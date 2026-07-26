// ==========================================================================
// runtime.cpp  —  End-to-end forward + generate (v1.0, minimal scope)
// ==========================================================================
// v1.0 implements a SIMPLE forward pass for smoke testing:
//   1. Look up tok_embeddings (fp16) for each input id.
//   2. Apply a single ternary matmul (the "lm_head" projection) to
//      produce logits over the vocabulary.
//   3. Apply sampling (greedy / top-k / top-p / temperature / rep pen).
//
// This is enough to validate the AVX-512 ternary matmul end-to-end and
// to provide a working CLI smoke test.  Full attention/MLP/RoPE/SwiGLU
// is v1.1.
// ==========================================================================

#include "ternair/ternair_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <string>


// ----------------------------------------------------------------------
// Lifecycle
// ----------------------------------------------------------------------

extern "C" TernairRuntime* ternair_create(void) {
    auto* rt = new (std::nothrow) TernairRuntime();
    if (!rt) return nullptr;
    rt->backend = TERNAIR_BACKEND_SCALAR;
    rt->rms_norm_eps = 1e-5f;
    return rt;
}

extern "C" void ternair_free(TernairRuntime* rt) {
    if (!rt) return;
    {
        // Stop workers first.
        std::lock_guard<std::mutex> lk(rt->mu);
        rt->stop.store(true);
    }
    rt->cv_task.notify_all();
    for (auto& t : rt->workers) {
        if (t.joinable()) t.join();
    }
    rt->workers.clear();
    delete rt;
}

extern "C" int ternair_ternary_matmul(
    const uint8_t* packed, int M, int Kp,
    const uint16_t* x_fp16_bits, int N,
    const float* gamma,
    uint16_t* out_fp16_bits
);

// Forward declaration of ternair_rmsnorm (defined in norm.cpp).
extern "C" void ternair_rmsnorm(
    uint16_t* x_fp16, int n,
    const uint16_t* weight_fp16, float eps
);


// ----------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------

extern "C" int ternair_forward(
    TernairRuntime* rt,
    const int32_t* input_ids, int input_len,
    float* logits_out
) {
    if (!rt || !input_ids || input_len <= 0 || !logits_out) return 1;
    if (rt->vocab_size == 0 || rt->hidden_size == 0) return 2;
    auto& m = rt->model;
    if (m.vocab_size == 0 || m.hidden_size == 0) return 2;

    // Take the LAST token's embedding as the input to lm_head.
    int last = input_ids[input_len - 1];
    if (last < 0 || last >= m.vocab_size) return 3;
    std::vector<uint16_t> h_fp16(static_cast<size_t>(m.hidden_size));
    std::memcpy(h_fp16.data(),
                m.embed_table.data() + static_cast<size_t>(last) * m.hidden_size,
                m.hidden_size * sizeof(uint16_t));

    // RMSNorm
    if (!m.final_norm.empty()) {
        ternair_rmsnorm(h_fp16.data(), m.hidden_size, m.final_norm.data(),
                        m.rms_norm_eps);
    }

    // LM head ternary matmul: (vocab, hidden) @ (hidden,) -> (vocab,)
    int Kp = m.lm_head.Kp;
    int M  = m.lm_head.M;
    int N  = m.lm_head.N;
    if (M == 0 || N == 0) return 4;
    std::vector<uint16_t> logits_fp16(static_cast<size_t>(M));
    int err = ternair_ternary_matmul(
        m.lm_head.packed.data(), M, Kp,
        h_fp16.data(), N,
        m.lm_head.gamma.data(),
        logits_fp16.data()
    );
    if (err) return 5;

    // Convert fp16 -> fp32
    for (int i = 0; i < M; ++i) {
        uint32_t sign = (uint32_t)(logits_fp16[i] >> 15) << 31;
        uint32_t exp  = (logits_fp16[i] >> 10) & 0x1F;
        uint32_t mant = logits_fp16[i] & 0x3FF;
        uint32_t fbits;
        if (exp == 0) fbits = sign;
        else if (exp == 31) fbits = sign | (0xFFu << 23) | (mant << 13);
        else fbits = sign | ((exp + 112u) << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &fbits, sizeof(f));
        logits_out[i] = f;
    }
    return 0;
}


// ----------------------------------------------------------------------
// Sampling
// ----------------------------------------------------------------------

static void softmax_inplace(std::vector<float>& v) {
    float max_v = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (auto& x : v) { x = std::exp(x - max_v); sum += x; }
    for (auto& x : v) x /= sum;
}

static int sample_top_k_top_p(
    const std::vector<float>& probs,
    int top_k, float top_p, float temperature,
    std::mt19937& rng
) {
    int V = static_cast<int>(probs.size());
    std::vector<std::pair<float, int>> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = {probs[i], i};
    std::sort(idx.begin(), idx.end(),
              [](auto& a, auto& b) { return a.first > b.first; });
    if (top_k > 0 && top_k < V) idx.resize(top_k);
    float total = 0.0f;
    for (auto& p : idx) total += p.first;
    if (total > 0.0f) {
        float cum = 0.0f;
        std::vector<std::pair<float, int>> kept;
        for (auto& p : idx) {
            cum += p.first / total;
            if (top_p > 0.0f && top_p < 1.0f && cum > top_p) break;
            kept.push_back(p);
        }
        idx = std::move(kept);
    }
    std::vector<float> weights;
    weights.reserve(idx.size());
    for (auto& p : idx) weights.push_back(p.first);
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return idx[dist(rng)].second;
}


// ----------------------------------------------------------------------
// Generate
// ----------------------------------------------------------------------

extern "C" int ternair_generate(
    TernairRuntime* rt,
    const int32_t* prompt, int prompt_len,
    int32_t* out_ids, int max_new_tokens,
    int eos_token_id,
    float temperature, int top_k, float top_p,
    float repetition_penalty
) {
    if (!rt || !prompt || prompt_len < 0 || !out_ids || max_new_tokens <= 0) return -1;
    int V = rt->vocab_size;
    if (V == 0) return -2;
    std::vector<int32_t> seq(prompt, prompt + prompt_len);
    std::mt19937 rng(42);

    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<float> logits(static_cast<size_t>(V));
        int err = ternair_forward(rt, seq.data(),
                                  static_cast<int>(seq.size()),
                                  logits.data());
        if (err) return -err;
        // Repetition penalty
        if (repetition_penalty != 1.0f) {
            for (int t : seq) {
                if (t < 0 || t >= V) continue;
                if (logits[t] > 0) logits[t] /= repetition_penalty;
                else             logits[t] *= repetition_penalty;
            }
        }
        int next;
        if (temperature <= 1e-6f) {
            next = static_cast<int>(std::distance(
                logits.begin(),
                std::max_element(logits.begin(), logits.end())));
        } else {
            std::vector<float> probs = logits;
            for (auto& x : probs) x /= temperature;
            softmax_inplace(probs);
            next = sample_top_k_top_p(probs, top_k, top_p, temperature, rng);
        }
        out_ids[prompt_len + step] = next;
        seq.push_back(next);
        if (eos_token_id >= 0 && next == eos_token_id) {
            return prompt_len + step + 1;
        }
    }
    return prompt_len + max_new_tokens;
}
