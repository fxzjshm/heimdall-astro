// third-party code should be used with their corresponding licenses
#pragma once

#include <boost/compute.hpp>

#include "hd/utils/external_function.dp.hpp"
#include "hd/utils/wrappers.dp.hpp"

#include "hd/types.h"

template <typename T, typename A1, typename A2>
void device_to_host_copy(boost::compute::vector<T, A2> &d,
                         std::vector<T, A1> &h) {
    h.resize(d.size());
    boost::compute::copy(d.begin(), d.end(), h.begin());
}

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

// see BOOST_COMPUTE_CLOSURE
#define BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(return_type, name, arguments, capture, source) \
    ::boost::compute::closure<                                                                  \
        return_type arguments, BOOST_TYPEOF(boost::tie capture)>                                \
        name =                                                                                  \
            ::boost::compute::detail::make_closure_impl<                                        \
                return_type arguments>(                                                         \
                #name, #arguments, boost::tie capture, #capture, source)

#define CL_TYPE_DEFINE(type) (std::string("typedef ") + boost::compute::type_name<type>() + " " + #type + ";")

inline std::string type_define_source() {
    return CL_TYPE_DEFINE(hd_byte) + CL_TYPE_DEFINE(hd_size) + CL_TYPE_DEFINE(hd_float);
}
