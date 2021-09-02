#pragma once

#include <boost/compute/algorithm/copy_if.hpp>
#include "hd/utils/transform_if.dp.hpp"

namespace boost {
namespace compute {

/// Copies each element in the range [\p first, \p last) for which
/// \p predicate returns \c true to the range beginning at \p result.
///
/// Space complexity: \Omega(2n)
template<class InputIterator1, class InputIterator2, class OutputIterator, class Predicate>
inline OutputIterator copy_if(InputIterator1 first,
                              InputIterator1 last,
                              InputIterator2 stencil,
                              OutputIterator result,
                              Predicate predicate,
                              command_queue &queue = system::default_queue())
{
    BOOST_STATIC_ASSERT(is_device_iterator<InputIterator1>::value);
    BOOST_STATIC_ASSERT(is_device_iterator<InputIterator2>::value);
    BOOST_STATIC_ASSERT(is_device_iterator<OutputIterator>::value);
    typedef typename std::iterator_traits<InputIterator1>::value_type T;

    return ::boost::compute::transform_if(
        first, last, stencil, result, identity<T>(), predicate, queue
    );
}

} // end compute namespace
} // end boost namespace