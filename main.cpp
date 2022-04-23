#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/spin_mutex.h>
#include "ticktock.h"
#include "pod.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
    [&] (tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            arr[i] = func(i);
        }
    });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
    [&] (tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            x[i] = a * x[i] + y[i];
        }
    });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, x.size()), (T)0,
    [&] (tbb::blocked_range<size_t> r, T local_ret) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_ret += x[i] * y[i];
        }
        return local_ret;
    }, [&] (T a, T b) {
        return a + b;
    });
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = x[0];
    std::mutex mtx;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
    [&] (tbb::blocked_range<size_t> r) {
        T local_min = x[r.begin()];
        for (size_t i = r.begin(); i < r.end(); i++) {
            if (x[i] < local_min)
                local_min = x[i];
        }
        std::lock_guard lck(mtx);
        if (local_min < ret)
            ret = local_min;
    });
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<pod<T>> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<pod<T>> res(2 * std::min(x.size(), y.size()));
    std::atomic<size_t> a_size = 0;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<pod<T>> local_res(2 * r.size());
        size_t lasize = 0;
        for (size_t i = r.begin(); i < r.end(); i++) {
            if (x[i] > y[i]) {
                local_res[lasize++] = x[i];
            } else if (y[i] > x[i] && y[i] > 0.5f) {
                local_res[lasize++] = y[i];
                local_res[lasize++] = x[i] * y[i];
            }
        }
        size_t base = a_size.fetch_add(lasize);
        for (size_t i = 0; i < lasize; i++) {
            res[base + i] = local_res[i];
        }
    });
    res.resize(a_size);
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), (T)0,
    [&] (tbb::blocked_range<size_t> r, T local_res, auto is_final) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_res += x[i];
            if (is_final)
                x[i] = local_res;
        }
        return local_res;
    }, [&] (T a, T b) {
        return a + b;
    });
    TOCK(scanner);
    return ret;
}

int main() {
    size_t n = 1<<26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    fill(x, [&] (size_t i) { return std::sin(i); });
    fill(y, [&] (size_t i) { return std::cos(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
