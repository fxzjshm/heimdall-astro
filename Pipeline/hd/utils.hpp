// third-party code should be used with their corresponding licenses
#pragma once

#include <boost/compute.hpp>

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

template <typename T, typename A1, typename A2>
void device_to_host_copy(boost::compute::vector<T, A2> &d,
          std::vector<T, A1> &h) {
    h.resize(d.size());
    boost::compute::copy(d.begin(), d.end(), h.begin());
}
