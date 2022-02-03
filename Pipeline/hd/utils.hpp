// third-party code should be used with their corresponding licenses
#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "dpct/dpl_extras/vector.h"
#include <sycl/execution_policy>

#include <type_traits>

#include <PRNG/MWC64X.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include "discard_iterator.hpp"
#include "permutation_iterator.hpp"

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

// global execution_policy
extern sycl::sycl_execution_policy<> execution_policy;

template <typename T, typename A1, typename A2>
void oneapi_copy(dpct::device_vector<T, A2> &d,
                 std::vector<T, A1> &h) {
    h.resize(d.size());
    auto queue = execution_policy.get_queue();
    dpct::copy(d.begin(), d.end(), h.begin(), queue);
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
        // wait here as operations above may be async, otherwise iterators may be invalid if memory is reallocated
        dpct::get_default_queue().wait();
        if (old_size < new_size) {
            ::sycl::impl::fill(execution_policy,
                dpct::device_vector<T, Allocator>::begin() + old_size, dpct::device_vector<T, Allocator>::begin() + new_size, x
            );
        }
    }
};

namespace sycl_pstl {
    using namespace sycl;
}

// https://stackoverflow.com/a/48458312 by smac89
// used in maximum_first functor for different situations
template <typename> struct is_tuple: std::false_type {};
template <typename ...T> struct is_tuple<std::tuple<T...>>: std::true_type {};

#if defined(SYCL_DEVICE_COPYABLE) && SYCL_DEVICE_COPYABLE
// patch for foreign iterators
template <typename T>
struct sycl::is_device_copyable<boost::iterators::counting_iterator<T>> : std::true_type {};
#endif