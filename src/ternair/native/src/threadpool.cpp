// ==========================================================================
// threadpool.cpp  —  Persistent worker thread pool
// ==========================================================================
// Workers spin on a condition_variable waiting for tasks.  This avoids
// the 10-50us cost of spawning std::thread per matmul, which would
// otherwise eat the 1ms latency budget.
// ==========================================================================

#include "ternair/ternair_internal.h"

#include <cstddef>

static void worker_main(TernairRuntime* rt) {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(rt->mu);
            rt->cv_task.wait(lk, [&] { return rt->stop.load() || !rt->tasks.empty(); });
            if (rt->stop.load() && rt->tasks.empty()) return;
            task = std::move(rt->tasks.front());
            rt->tasks.pop();
            rt->n_active++;
        }
        task();
        {
            std::lock_guard<std::mutex> lk(rt->mu);
            rt->n_active--;
            if (rt->tasks.empty() && rt->n_active == 0) {
                rt->cv_done.notify_all();
            }
        }
    }
}

void ternair_pool_init(TernairRuntime* rt, int num_threads) {
    if (num_threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        num_threads = hc > 0 ? static_cast<int>(hc) : 1;
    }
    rt->n_threads = num_threads;
    if (rt->n_threads <= 1) return;  // no pool needed
    rt->workers.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        rt->workers.emplace_back(worker_main, rt);
    }
}

void ternair_pool_submit(TernairRuntime* rt, int n_tasks,
                          const std::function<void(int)>& body) {
    if (rt->n_threads <= 1 || n_tasks <= 1) {
        for (int i = 0; i < n_tasks; ++i) body(i);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(rt->mu);
        for (int i = 0; i < n_tasks; ++i) {
            rt->tasks.push([body, i] { body(i); });
        }
    }
    rt->cv_task.notify_all();
    std::unique_lock<std::mutex> lk(rt->mu);
    rt->cv_done.wait(lk, [&] { return rt->tasks.empty() && rt->n_active == 0; });
}
