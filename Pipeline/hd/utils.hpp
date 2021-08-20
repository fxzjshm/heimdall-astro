// third-party code should be used with their corresponding licenses
#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

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

// copied from oneapi/dpl/pstl/iterator_impl.h, with assert removed
template <typename _Ip>
class counting_iterator {

public:
    typedef typename try_make_signed<_Ip>::type difference_type;
    typedef _Ip value_type;
    typedef const _Ip *pointer;
    // There is no storage behind the iterator, so we return a value instead of
    // reference.
    typedef _Ip reference;
    typedef ::std::random_access_iterator_tag iterator_category;
    using is_passed_directly = ::std::true_type;

    counting_iterator() : __my_counter_() {}
    explicit counting_iterator(_Ip __init) : __my_counter_(__init) {}

    reference operator*() const { return __my_counter_; }
    reference operator[](difference_type __i) const { return *(*this + __i); }

    difference_type operator-(const counting_iterator &__it) const {
        return __my_counter_ - __it.__my_counter_;
    }

    counting_iterator &operator+=(difference_type __forward) {
        __my_counter_ += __forward;
        return *this;
    }
    counting_iterator &operator-=(difference_type __backward) {
        return *this += -__backward;
    }
    counting_iterator &operator++() { return *this += 1; }
    counting_iterator &operator--() { return *this -= 1; }

    counting_iterator operator++(int) {
        counting_iterator __it(*this);
        ++(*this);
        return __it;
    }
    counting_iterator operator--(int) {
        counting_iterator __it(*this);
        --(*this);
        return __it;
    }

    counting_iterator operator-(difference_type __backward) const {
        return counting_iterator(__my_counter_ - __backward);
    }
    counting_iterator operator+(difference_type __forward) const {
        return counting_iterator(__my_counter_ + __forward);
    }
    friend counting_iterator operator+(difference_type __forward,
                                       const counting_iterator __it) {
        return __it + __forward;
    }

    bool operator==(const counting_iterator &__it) const {
        return *this - __it == 0;
    }
    bool operator!=(const counting_iterator &__it) const {
        return !(*this == __it);
    }
    bool operator<(const counting_iterator &__it) const {
        return *this - __it < 0;
    }
    bool operator>(const counting_iterator &__it) const { return __it < *this; }
    bool operator<=(const counting_iterator &__it) const {
        return !(*this > __it);
    }
    bool operator>=(const counting_iterator &__it) const {
        return !(*this < __it);
    }

private:
    _Ip __my_counter_;
};
} // namespace third_party

// reference:
// https://thrust.github.io/doc/group__scattering_ga72c5ec1e36f08a1bd7b4e0b20e7e906d.html
template <typename InputIterator1, typename InputIterator2,
          typename InputIterator3, typename RandomAccessIterator>
void scatter_if(InputIterator1 first, InputIterator1 last, InputIterator2 map,
                InputIterator3 stencil, RandomAccessIterator output) {
#pragma unroll
    for (auto i = first; i != last; ++i) {
        if (*(stencil + (i - first))) {
            output[*(map + (i - first))] = (*i);
        }
    }
    /*
   std::for_each(
       oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
       first, last, [=](const auto& t) {
         auto i = InputIterator1(&t);
         if (*(stencil + (i - first))) {
           output[*(map + (i - first))] = (*i);
         }
       });
   */
}

// reference:
// https://thrust.github.io/doc/group__gathering_ga6fdb1fe3ff0d9ce01f41a72fa94c56df.html
template <typename InputIterator, typename RandomAccessIterator,
          typename OutputIterator>
void gather(InputIterator map_first, InputIterator map_last,
                      RandomAccessIterator input_first, OutputIterator result) {

#pragma unroll
    for (auto i = map_first; i != map_last; i++) {
        *(result + (i - map_first)) = input_first[*i];
    }
    /*
    std::for_each(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        map_first, map_last, [=](auto& t) {
          auto i = InputIterator1(&t);
          *(result + (i - map_first)) = input_first[*i];
        });
  */
}

namespace third_party {
// reference:
// https://github.com/NVIDIA/thrust/blob/fa54f2c6f1217237953f27ddf67f901b6b34fbdd/thrust/system/detail/sequential/reduce_by_key.h
template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator1, typename OutputIterator2,
          typename BinaryPredicate, typename BinaryFunction>
std::pair<OutputIterator1, OutputIterator2>
reduce_by_key(InputIterator1 keys_first, InputIterator1 keys_last,
              InputIterator2 values_first, OutputIterator1 keys_output,
              OutputIterator2 values_output, BinaryPredicate binary_pred,
              BinaryFunction binary_op) {
    typedef
        typename std::iterator_traits<InputIterator1>::value_type InputKeyType;
    typedef
        typename std::iterator_traits<InputIterator2>::value_type InputValueType;

    // Use the input iterator's value type per https://wg21.link/P0571
    using TemporaryType =
        typename std::iterator_traits<InputIterator2>::value_type;

    if (keys_first != keys_last) {
        InputKeyType temp_key = *keys_first;
        TemporaryType temp_value = *values_first;

        for (++keys_first, ++values_first; keys_first != keys_last;
             ++keys_first, ++values_first) {
            InputKeyType key = *keys_first;
            InputValueType value = *values_first;

            if (binary_pred(temp_key, key)) {
                temp_value = binary_op(temp_value, value);
            } else {
                *keys_output = temp_key;
                *values_output = temp_value;

                ++keys_output;
                ++values_output;

                temp_key = key;
                temp_value = value;
            }
        }

        *keys_output = temp_key;
        *values_output = temp_value;

        ++keys_output;
        ++values_output;
    }

    return std::make_pair(keys_output, values_output);
}
} // namespace third_party

template <typename device_vector_type, typename host_vector_type>
void oneapi_copy(device_vector_type device_vector,
                 host_vector_type host_vector) {
    std::copy(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        device_vector.begin(), device_vector.end(), host_vector.begin());
}
