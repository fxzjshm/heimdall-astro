#include <boost/compute.hpp>

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