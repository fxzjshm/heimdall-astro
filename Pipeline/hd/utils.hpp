// third-party code should be used with their corresponding licenses
#pragma once

#include <boost/compute.hpp>

#include <type_traits>

#include "hd/types.h"

template <typename T, typename A1, typename A2>
void device_to_host_copy(boost::compute::vector<T, A2> &d,
                         std::vector<T, A1> &h) {
    h.resize(d.size());
    boost::compute::copy(d.begin(), d.end(), h.begin());
}

template <class T, class Alloc = boost::compute::buffer_allocator<T>>
class device_vector_wrapper : public boost::compute::vector<T, Alloc> {
public:
    using boost::compute::vector<T, Alloc>::vector;

    template <typename OtherAllocator>
    boost::compute::vector<T, Alloc> &operator=(const std::vector<T, OtherAllocator> &v) {
        return boost::compute::vector<T, Alloc>::operator=(v);
    }

    void resize(size_type new_size, const T &x = T()) {
        size_type old_size = boost::compute::vector<T, Alloc>::size();
        boost::compute::vector<T, Alloc>::resize(new_size, x);
        boost::compute::iota(boost::compute::vector<T, Alloc>::begin() + old_size, boost::compute::vector<T, Alloc>::end(), T);
    }
};

/*

template<class T>
class buffer_iterator_wrapper;
template<class T>
class device_ptr_wrapper;

template<class T>
class device_ptr_wrapper : public boost::compute::detail::device_ptr<T> {
public:
    using boost::compute::detail::device_ptr<T>::device_ptr;

    operator boost::compute::buffer_iterator<T>() {
        return boost::compute::buffer_iterator<T>(boost::compute::detail::device_ptr<T>::get_buffer(),
                                          boost::compute::detail::device_ptr<T>::get_index());
    }
};

template<class T>
class buffer_iterator_wrapper : public boost::compute::buffer_iterator<T> {
public:
    using boost::compute::buffer_iterator<T>::buffer_iterator;
    
    buffer_iterator_wrapper(boost::compute::buffer_iterator<T> iter)
        : boost::compute::buffer_iterator<T>::buffer_iterator(iter) {}
    
    operator boost::compute::detail::device_ptr<T>() {
        return boost::compute::detail::device_ptr<T>(boost::compute::buffer_iterator<T>::get_buffer(),
                                     boost::compute::buffer_iterator<T>::get_index());
    }
};

*/

namespace boost {
namespace compute {
namespace detail {

template <class T>
struct set_kernel_arg<buffer_iterator<T>> {
    void operator()(kernel &kernel_, size_t index, const buffer_iterator<T> &iter) {
        kernel_.set_arg(index, iter.get_buffer());
    }
};

template <class T>
struct capture_traits<buffer_iterator<T>> {
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

class external_function {
public:
    std::string name;
    std::string source;
    std::map<std::string, std::string> definitions;

    external_function(std::string name_, std::string source_, std::map<std::string, std::string> definitions_ = std::map<std::string, std::string>())
        : name(name_), source(source_), definitions(definitions_) {}
};

template <typename T>
class invoked_function_with_external_function;

template <typename T>
class function_with_external_function {
public:
    typedef typename T::result_type result_type;

    T main_func;
    std::vector<external_function> funcs;
    // std::vector<std::string> type_defines;

    function_with_external_function(T main_func_, external_function func_/*, std::vector<std::string> type_defines_ = std::vector<std::string>()*/)
        : function_with_external_function(main_func_, {func_}, /*type_defines_*/) {}

    function_with_external_function(T main_func_, std::vector<external_function> funcs_/*, std::vector<std::string> type_defines_ = std::vector<std::string>()*/)
        : main_func(main_func_), funcs(funcs_)/*, type_defines(type_defines_)*/ {}

    /// \internal_
    template<class Arg1>
    auto operator()(const Arg1 &arg1) const {
        return invoked_function_with_external_function(main_func(arg1), funcs/*, type_defines*/);
    }

    /// \internal_
    template<class Arg1, class Arg2>
    auto operator()(const Arg1 &arg1, const Arg2 &arg2) const {
        return invoked_function_with_external_function(main_func(arg1, arg2), funcs/*, type_defines*/);
    }

    /// \internal_
    template<class Arg1, class Arg2, class Arg3>
    auto operator()(const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3) const {
        return invoked_function_with_external_function(main_func(arg1, arg2, arg3), funcs/*, type_defines*/);
    }
};

template <typename T>
class invoked_function_with_external_function {
public:
    std::vector<external_function> funcs;
    T main_func;
    // std::vector<std::string> type_defines;

    invoked_function_with_external_function(T main_func_, std::vector<external_function> funcs_/*, std::vector<std::string> type_defines_*/)
        : main_func(main_func_), funcs(funcs_)/*, type_defines(type_defines_)*/ {}
};

namespace boost {
namespace compute {
namespace detail {

inline meta_kernel &operator<<(meta_kernel &kernel, const external_function &func) {
    if (func.definitions.size()) {
        kernel.add_function(func.name, func.source, func.definitions);
    } else {
        kernel.add_function(func.name, func.source); // for cache
    }
    return kernel;
}

template <class T>
inline meta_kernel &operator<<(meta_kernel &kernel, const invoked_function_with_external_function<T> &func) {
    /*
    std::string type_declarations;
    for(std::string s : func.type_defines) {
        type_declarations += s;
    }
    kernel.add_type_declaration<int>(type_declarations); // HACK
    */
    for(external_function f : func.funcs) {
        kernel << f;
    }
    kernel << func.main_func;
    return kernel;
}

} // namespace detail
} // namespace compute
} // namespace boost

#define DEFINE_BOTH_SIDE(name, source) \
    source; \
    const external_function name##_function(#name, #source);