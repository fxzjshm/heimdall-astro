#pragma once

#include <boost/compute/iterator/buffer_iterator.hpp>

namespace boost {
namespace compute {
namespace detail {

template <class T>
struct set_kernel_arg<buffer_iterator<T> > {
    void operator()(kernel &kernel_, size_t index, const buffer_iterator<T> &iter) {
        kernel_.set_arg(index, iter.get_buffer());
    }
};

template <class T>
struct capture_traits<buffer_iterator<T> > {
    static std::string type_name() {
        return std::string("__global ") + ::boost::compute::type_name<T>() + "*";
    }
};

template <class T>
inline meta_kernel &operator<<(meta_kernel &kernel, const buffer_iterator<T> &iter) {
    return kernel << kernel.get_buffer_identifier<T>(iter.get_buffer()) << '+' << iter.get_index();
}

} // namespace detail
} // namespace compute
} // namespace boost