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
#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include "discard_iterator.hpp"
#include "permutation_iterator.hpp"
#include <sycl/container/device_vector.hpp>
#include <sycl/helpers/sycl_usm_vector.hpp>

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

template <typename T, sycl::usm::alloc AllocKind = sycl::usm::alloc::device,
          size_t align = sizeof(T)>
class device_allocator {
public:
  device_allocator(cl::sycl::queue &queue_) : queue(queue_){};

  T *allocate(std::size_t num_elements) {
    T *ptr = sycl::aligned_alloc_device<T>(align, num_elements, queue);
    if (!ptr)
      throw std::runtime_error("device_allocator: Allocation failed");
    return ptr;
  }

  void deallocate(T *ptr, std::size_t size) {
    if (ptr)
      sycl::free(ptr, queue);
  }

private:
  cl::sycl::queue queue;
};

template <typename T, sycl::usm::alloc AllocKind = sycl::usm::alloc::shared,
          size_t align = sizeof(T)>
class usm_device_allocator : public sycl::usm_allocator<T, AllocKind, align> {
  using Base = sycl::usm_allocator<T, AllocKind, align>;

public:
  usm_device_allocator(cl::sycl::queue &queue_) : Base(queue_), queue(queue_){};

  T *allocate(std::size_t num_elements) {
    T *ptr = Base::allocate(num_elements);
    mem_advise(ptr, num_elements * sizeof(T), queue);
    return ptr;
  }

private:
  cl::sycl::queue queue;

  // backend specific
  inline void mem_advise(const void* ptr, size_t num_bytes, sycl::queue& queue) {
#if defined(SYCL_IMPLEMENTATION_ONEAPI)

#if defined(SYCL_EXT_ONEAPI_BACKEND_CUDA)
    return queue.mem_advise(ptr, num_bytes, PI_MEM_ADVICE_CUDA_SET_PREFERRED_LOCATION);
#else
#warning "Detected OneAPI but no known advice. (TODO)"
    return;
#endif // defined(SYCL_EXT_ONEAPI_BACKEND_CUDA)

#elif defined(__HIPSYCL__)

#if defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
    return sycl::mem_advise(ptr, num_bytes, cudaMemAdviseSetPreferredLocation, queue);
#elif defined(__HIPSYCL_ENABLE_HIP_TARGET__)
    return sycl::mem_advise(ptr, num_bytes, hipMemAdviseSetPreferredLocation, queue);
#elif defined(__HIPSYCL_ENABLE_OMPHOST_TARGET__)
    return;
#else
#warning "Detected hipSYCL but no known advice. (TODO)"
    return;
#endif

#else
#warning "Unknown backend for advising USM allocator to use device memory, performance may suffer."
    return;
#endif
  }
};

//#define USE_DPCT 1

#ifdef USE_DPCT

#ifndef DPCT_USM_LEVEL_NONE
template <typename T,
          typename Allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared> >
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
            execution_policy.get_queue().wait();
        }
    }
};

namespace heimdall {
namespace util {

template <typename T>
using device_iterator = dpct::device_iterator<T>;

template <typename T>
using device_pointer = dpct::device_pointer<T>;

template <typename... Args>
inline decltype(auto) get_raw_pointer(Args &&...args) {
  return dpct::get_raw_pointer(std::forward<Args>(args)...);
}


} // namespace util
} // namespace heimdall

#else

// Switch between implementations
template <typename T, typename Allocator = /*sycl::helpers::usm_allocator<T>*/  sycl::helpers::device_allocator<T> >
using device_vector_wrapper = sycl::impl::device_vector<T, Allocator>;

namespace heimdall {
namespace util {

template <typename T>
using device_iterator = sycl::helpers::device_iterator<T>;

template <typename T>
using device_pointer = sycl::helpers::device_pointer<T>;

template <typename... Args>
inline decltype(auto) get_raw_pointer(Args &&...args) {
  return sycl::helpers::get_raw_pointer(std::forward<Args>(args)...);
}

} // namespace util
} // namespace heimdall

#endif

namespace heimdall {
namespace util {


template <typename T, typename A2, typename OutputIterator>
inline void copy(const device_vector_wrapper<T, A2> &d, OutputIterator out) {
  static_assert(std::is_convertible_v<decltype(&(*out)), T*>);
  auto queue = execution_policy.get_queue();
  queue.copy(heimdall::util::get_raw_pointer(&d[0]), &(*out), d.size()).wait();
}

template <typename T, typename A1, typename A2>
inline void copy(const device_vector_wrapper<T, A2> &d, std::vector<T, A1> &h) {
  h.resize(d.size());
  copy(d, h.begin());
}

template <typename T, typename OutputIterator>
inline void copy(const device_pointer<T> d_in_first, const device_pointer<T> d_in_last, OutputIterator h_out) {
  static_assert(std::is_convertible_v<decltype(&(*h_out)), T*>);
  auto size = std::distance(d_in_first, d_in_last);
  auto queue = execution_policy.get_queue();
  queue.copy(heimdall::util::get_raw_pointer(d_in_first), &(*h_out), size).wait();
}

} // namespace util
} // namespace heimdall

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
template <class ElementIterator, class IndexIterator>
struct sycl::is_device_copyable<boost::iterators::permutation_iterator<ElementIterator, IndexIterator>, std::enable_if_t<!std::is_trivially_copyable<boost::iterators::permutation_iterator<ElementIterator, IndexIterator>>::value>> : std::true_type {};
#endif