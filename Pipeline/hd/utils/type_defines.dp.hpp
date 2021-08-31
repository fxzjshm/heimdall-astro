#pragma once

#include "hd/types.h"
#include <string>
#include <boost/compute/type_traits/type_name.hpp>

#define CL_TYPE_DEFINE(type) (std::string("typedef ") + boost::compute::type_name<type>() + " " + #type + ";")

inline std::string type_define_source() {
    return CL_TYPE_DEFINE(hd_byte) + CL_TYPE_DEFINE(hd_size) + CL_TYPE_DEFINE(hd_float);
}
