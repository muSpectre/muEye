/**
 * @file   parallel_for.hh
 *
 * @brief  Minimal parallel-for over an index range.
 *
 * Uses OpenMP when the compiler provides it; otherwise falls back to a
 * std::thread split so the CPU renderer is genuinely parallel even on toolchains
 * without OpenMP (e.g. Apple clang).
 *
 * Part of muEye, a viewer for muGrid data.
 */

#ifndef MUEYE_PARALLEL_FOR_HH_
#define MUEYE_PARALLEL_FOR_HH_

#include <algorithm>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace mueye {

/**
 * Invoke @p body(i) for i in [0, n). @p nb_threads <= 0 means "auto".
 *
 * @p body must be safe to call concurrently for distinct i.
 */
template <class Body>
void parallel_for(int n, int nb_threads, Body &&body) {
  if (n <= 0) return;

#if defined(_OPENMP)
  if (nb_threads > 0) omp_set_num_threads(nb_threads);
#pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < n; ++i) body(i);
#else
  unsigned hw = nb_threads > 0
                    ? static_cast<unsigned>(nb_threads)
                    : std::thread::hardware_concurrency();
  if (hw == 0) hw = 4;
  unsigned nt = std::min<unsigned>(hw, static_cast<unsigned>(n));
  if (nt <= 1) {
    for (int i = 0; i < n; ++i) body(i);
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve(nt);
  int chunk = (n + static_cast<int>(nt) - 1) / static_cast<int>(nt);
  for (unsigned t = 0; t < nt; ++t) {
    int begin = static_cast<int>(t) * chunk;
    int end = std::min(begin + chunk, n);
    if (begin >= end) break;
    pool.emplace_back([begin, end, &body]() {
      for (int i = begin; i < end; ++i) body(i);
    });
  }
  for (auto &th : pool) th.join();
#endif
}

}  // namespace mueye

#endif  // MUEYE_PARALLEL_FOR_HH_
