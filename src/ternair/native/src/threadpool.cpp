/**
 * threadpool.cpp — Minimal work-stealing thread pool
 *
 * Used to parallelise the outer (output row) loop of matmul_dispatch
 * across physical cores. Falls back to single-thread when n_threads == 1.
 */
#include "ternair_native.h"
#include <thread>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstdint>

namespace ternair {

struct ThreadPool {
    explicit ThreadPool(int n) {
        n = (n <= 0) ? (int)std::thread::hardware_concurrency() : n;
        if (n < 1) n = 1;
        n_threads = n;
        stop = false;
        for (int i = 0; i < n; ++i)
            workers.emplace_back([this]{ worker_loop(); });
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lk(mtx); stop = true; }
        cv.notify_all();
        for (auto& t : workers) t.join();
    }

    void submit(std::function<void()> fn) {
        { std::unique_lock<std::mutex> lk(mtx); tasks.push(std::move(fn)); }
        cv.notify_one();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lk(done_mtx);
        done_cv.wait(lk, [this]{ return pending.load() == 0; });
    }

    void parallel_for(int n, std::function<void(int)> fn) {
        if (n_threads == 1 || n <= 1) { for (int i=0;i<n;++i) fn(i); return; }
        pending.store(n);
        for (int i = 0; i < n; ++i) {
            submit([this, i, fn]{
                fn(i);
                if (pending.fetch_sub(1) == 1) done_cv.notify_one();
            });
        }
        wait_all();
    }

    int n_threads;
private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            { std::unique_lock<std::mutex> lk(mtx);
              cv.wait(lk, [this]{ return stop || !tasks.empty(); });
              if (stop && tasks.empty()) return;
              task = std::move(tasks.front()); tasks.pop(); }
            task();
        }
    }
    std::vector<std::thread>        workers;
    std::queue<std::function<void()>> tasks;
    std::mutex                       mtx, done_mtx;
    std::condition_variable          cv, done_cv;
    std::atomic<int>                 pending{0};
    bool                             stop;
};

static ThreadPool* g_pool = nullptr;
static std::mutex  g_pool_mtx;

void threadpool_init(int n_threads) {
    std::lock_guard<std::mutex> lk(g_pool_mtx);
    delete g_pool;
    g_pool = new ThreadPool(n_threads);
}

void parallel_for(int n, std::function<void(int)> fn) {
    if (!g_pool) threadpool_init(0);
    g_pool->parallel_for(n, fn);
}

int threadpool_n_threads() {
    if (!g_pool) return (int)std::thread::hardware_concurrency();
    return g_pool->n_threads;
}

} // namespace ternair
