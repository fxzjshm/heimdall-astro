#pragma once

#include "hd/utils/meta_kernel.dp.hpp"
#include "hd/utils/buffer_iterator.dp.hpp"
#include <boost/compute/container/vector.hpp>

template <typename T, typename A1, typename A2>
void device_to_host_copy(boost::compute::vector<T, A2> &d,
                         std::vector<T, A1> &h) {
    h.resize(d.size());
    boost::compute::copy(d.begin(), d.end(), h.begin());
    boost::compute::system::default_queue().finish();
}