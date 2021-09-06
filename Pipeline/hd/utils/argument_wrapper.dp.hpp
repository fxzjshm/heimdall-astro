#pragma once

#include <string>
#include <boost/preprocessor/facilities/overload.hpp>
#include "hd/utils/meta_kernel.dp.hpp"

/// this class stores some functor parameters and use them as kernel arguments instead of literals,
/// so more kernel cache would be hit
template<typename T>
class argument_wrapper {
public:
    std::string name;
    T val;

    argument_wrapper(std::string name_, T val_ = T())
        : name(name_), val(val_) {}

    argument_wrapper<T>& operator=(T& other) {
        val = other;
    }

    /// as if this is a constructor
    void operator()(T val_) {
        val = val_;
    }
};

// use these in class constructor
#define WRAP_ARG_1(var) var(#var, var)
#define WRAP_ARG_2(var, val) var(#var, val)

#if !BOOST_PP_VARIADICS_MSVC
#define WRAP_ARG(...) BOOST_PP_OVERLOAD(WRAP_ARG_,__VA_ARGS__)(__VA_ARGS__)
#else
#define WRAP_ARG(...) \
  BOOST_PP_CAT(BOOST_PP_OVERLOAD(WRAP_ARG_,__VA_ARGS__)(__VA_ARGS__),BOOST_PP_EMPTY())
#endif

namespace boost {
namespace compute {
namespace detail {

template <class T>
struct set_kernel_arg<argument_wrapper<T> > {
    void operator()(kernel &kernel_, size_t index, const argument_wrapper<T> &wrapper) {
        kernel_.set_arg(index, wrapper.val);
    }
};

template <class T>
struct capture_traits<argument_wrapper<T> > {
    static std::string type_name() {
        return ::boost::compute::type_name<T>();
    }
};

template <class T>
inline meta_kernel &operator<<(meta_kernel &kernel, const argument_wrapper<T> &wrapper) {
    kernel.add_set_arg(wrapper.name, wrapper.val);
    return kernel << wrapper.name;
}

} // namespace detail
} // namespace compute
} // namespace boost