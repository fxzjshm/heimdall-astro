// third-party code should be used with their corresponding licenses
#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_extras/vector.h>
#include <dpct/dpl_extras/algorithm.h>

#include <PRNG/MWC64X.hpp>
#include <type_traits>

typedef prng::mwc64x_32 random_engine;
template <typename T>
struct uniform_int_distribution {
    T min_value;
    T max_value;

    uniform_int_distribution(T m, T M) : min_value(m), max_value(M) {}

    T operator()(random_engine rng) {
        return (rng() % (max_value - min_value)) + min_value;
    }
};

namespace third_party {

// std::make_signed that accepts float from
// https://stackoverflow.com/questions/16377736/stdmake-signed-that-accepts-floating-point-types
template <typename T>
struct identity { using type = T; };

template <typename T>
using try_make_signed =
    typename std::conditional<std::is_integral<T>::value, std::make_signed<T>,
                              identity<T>>::type;

} // namespace third_party

template <class ExecutionPolicy, class InputIterator, class MapIterator,
          class StencilIterator, class OutputIterator, class Predicate>
void scatter_if(ExecutionPolicy&& exec, InputIterator first, InputIterator last,
                MapIterator map, StencilIterator stencil, OutputIterator result,
                Predicate predicate) {
  dpct::transform_if(exec, first, last, stencil, oneapi::dpl::make_permutation_iterator(result, map), std::identity(), predicate);
}

template <class ExecutionPolicy, class InputIterator, class MapIterator,
          class StencilIterator, class OutputIterator>
void scatter_if(ExecutionPolicy&& exec, InputIterator first, InputIterator last,
                MapIterator map, StencilIterator stencil, OutputIterator result) {
  ::scatter_if(exec, first, last, map, stencil, result, std::identity());
}

template <typename T, typename A1, typename A2>
void oneapi_copy(dpct::device_vector<T, A2> &d,
                 std::vector<T, A1> &h) {
    h.resize(d.size());

    std::copy(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        d.begin(), d.end(), h.begin());

    // dpct::get_default_queue().memcpy(&h[0], d.begin(), d.size() * sizeof(T)).wait();
}

#ifndef DPCT_USM_LEVEL_NONE
template <typename T,
          typename Allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>>
#else
template <typename T, typename Allocator = cl::sycl::buffer_allocator>
#endif
class device_vector_wrapper : public dpct::device_vector<T, Allocator> {
public:
    using size_type = std::size_t;
    using dpct::device_vector<T, Allocator>::device_vector;

    template <typename OtherAllocator>
    dpct::device_vector<T, Allocator> &operator=(const std::vector<T, OtherAllocator> &v) {
        return dpct::device_vector<T, Allocator>::operator=(v);
    }

    void resize(size_type new_size, const T &x = T()) {
        size_type old_size = dpct::device_vector<T, Allocator>::size();
        dpct::device_vector<T, Allocator>::resize(new_size, x);
        if (old_size < new_size) {
            std::fill(
                oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                dpct::device_vector<T, Allocator>::begin() + old_size, dpct::device_vector<T, Allocator>::begin() + new_size, x
            );
        }
    }
};