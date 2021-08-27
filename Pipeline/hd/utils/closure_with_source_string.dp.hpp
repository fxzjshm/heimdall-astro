#pragma once

#include <boost/compute/closure.hpp>

// see BOOST_COMPUTE_CLOSURE
#define BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(return_type, name, arguments, capture, source) \
    ::boost::compute::closure<                                                                  \
        return_type arguments, BOOST_TYPEOF(boost::tie capture)>                                \
        name =                                                                                  \
            ::boost::compute::detail::make_closure_impl<                                        \
                return_type arguments>(                                                         \
                #name, #arguments, boost::tie capture, #capture, source)
