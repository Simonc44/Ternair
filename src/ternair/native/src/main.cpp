// ==========================================================================
// main.cpp  —  CLI entry point for the Ternair native runtime
// ==========================================================================
// Usage:
//   ternair_native --model model.safetensors \
//                   --prompt-tokens 1,544,12  \
//                   --max-tokens 16           \
//                   --threads 0               \
//                   --temperature 0.7         \
//                   --top-k 40                \
//                   --top-p 0.9               \
//                   --eos 2
//
// v1.0: minimal end-to-end demo.  For real tokenization, use the
// Python ctypes wrapper which integrates with transformers / sentencepiece.
// ==========================================================================

#include "ternair/ternair_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>

static std::vector<int> parse_int_csv(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            try { out.push_back(std::stoi(item)); }
            catch (...) { fprintf(stderr, "bad prompt-tokens: '%s'\n", item.c_str()); }
        }
    }
    return out;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <file.safetensors> [options]\n"
        "  --model <path>             SafeTensors file (required)\n"
        "  --prompt-tokens 1,2,3      Initial tokens (default: 0,1,2)\n"
        "  --max-tokens  N            New tokens to generate (default 16)\n"
        "  --threads  N               Worker threads (0=auto, default 0)\n"
        "  --temperature  T           Sampling T (0=greedy, default 0.7)\n"
        "  --top-k  K                 Top-K filtering (0=off, default 40)\n"
        "  --top-p  P                 Top-P filtering (0=off, default 0.9)\n"
        "  --eos  ID                  EOS token id (-1=off, default -1)\n"
        "  --info                     Print runtime info and exit\n",
        prog);
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    std::string prompt_csv = "0,1,2";
    int max_tokens = 16;
    int threads = 0;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    int eos = -1;
    bool info_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--prompt-tokens" && i + 1 < argc) prompt_csv = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--temperature" && i + 1 < argc) temperature = std::atof(argv[++i]);
        else if (a == "--top-k" && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (a == "--top-p" && i + 1 < argc) top_p = std::atof(argv[++i]);
        else if (a == "--eos" && i + 1 < argc) eos = std::atoi(argv[++i]);
        else if (a == "--info") info_only = true;
        else { usage(argv[0]); return 1; }
    }
    if (!model_path) { usage(argv[0]); return 1; }

    TernairRuntime* rt = ternair_create();
    if (!rt) { fprintf(stderr, "ternair_create failed\n"); return 2; }
    int err = ternair_load(rt, model_path, threads);
    if (err) { fprintf(stderr, "ternair_load failed: %d (%s)\n",
                       err, model_path); ternair_free(rt); return 3; }

    printf("[ternair_native] backend=%s layers=%d hidden=%d vocab=%d\n",
           ternair_backend_name(ternair_backend(rt)),
           ternair_num_layers(rt),
           ternair_hidden_size(rt),
           ternair_vocab_size(rt));

    if (info_only) { ternair_free(rt); return 0; }

    std::vector<int32_t> prompt;
    for (int t : parse_int_csv(prompt_csv)) prompt.push_back(static_cast<int32_t>(t));
    std::vector<int32_t> out(static_cast<size_t>(prompt.size() + max_tokens));

    int n = ternair_generate(
        rt,
        prompt.data(), static_cast<int>(prompt.size()),
        out.data(), max_tokens,
        eos, temperature, top_k, top_p,
        /*repetition_penalty=*/1.0f
    );
    if (n < 0) { fprintf(stderr, "ternair_generate failed: %d\n", n);
                  ternair_free(rt); return 4; }

    printf("[output] tokens (%d):", n);
    for (int i = 0; i < n; ++i) printf(" %d", out[i]);
    printf("\n");

    ternair_free(rt);
    return 0;
}
