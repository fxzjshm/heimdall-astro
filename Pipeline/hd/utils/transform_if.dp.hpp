// edited from boost/compute/algorithm/transform_if.hpp
// to add the `stencil` parameter
#pragma once

#include <boost/compute/algorithm/transform_if.hpp>

namespace boost {
namespace compute {
namespace detail {

// Space complexity: O(2n)
template<class InputIterator1, class InputIterator2, class OutputIterator, class UnaryFunction, class Predicate>
inline OutputIterator transform_if_impl(InputIterator1 first,
                                        InputIterator1 last,
                                        InputIterator2 stencil,
                                        OutputIterator result,
                                        UnaryFunction function,
                                        Predicate predicate,
                                        bool copyIndex,
                                        command_queue &queue)
{
    typedef typename std::iterator_traits<OutputIterator>::difference_type difference_type;

    size_t count = detail::iterator_range_size(first, last);
    if(count == 0){
        return result;
    }

    const context &context = queue.get_context();

    // storage for destination indices
    ::boost::compute::vector<cl_uint> indices(count, context);

    // write counts
    ::boost::compute::detail::meta_kernel k1("transform_if_write_counts");
    k1 << indices.begin()[k1.get_global_id(0)] << " = "
           << predicate(stencil[k1.get_global_id(0)]) << " ? 1 : 0;\n";
    k1.exec_1d(queue, 0, count);

    // scan indices
    size_t copied_element_count = (indices.cend() - 1).read(queue);
    ::boost::compute::exclusive_scan(
        indices.begin(), indices.end(), indices.begin(), queue
    );
    copied_element_count += (indices.cend() - 1).read(queue); // last scan element plus last mask element

    // copy values
    ::boost::compute::detail::meta_kernel k2("transform_if_do_copy");
    k2 << "if(" << predicate(stencil[k2.get_global_id(0)]) << ")" <<
          "    " << result[indices.begin()[k2.get_global_id(0)]] << "=";

    if(copyIndex){
        k2 << k2.get_global_id(0) << ";\n";
    }
    else {
        k2 << function(first[k2.get_global_id(0)]) << ";\n";
    }

    k2.exec_1d(queue, 0, count);

    return result + static_cast<difference_type>(copied_element_count);
}

} // end detail namespace

/// Copies each element in the range [\p first, \p last) for which
/// \p predicate returns \c true to the range beginning at \p result.
///
/// Space complexity: O(2n)
template<class InputIterator1, class InputIterator2, class OutputIterator, class UnaryFunction, class Predicate>
inline OutputIterator transform_if(InputIterator1 first,
                                   InputIterator1 last,
                                   InputIterator2 stencil,
                                   OutputIterator result,
                                   UnaryFunction function,
                                   Predicate predicate,
                                   command_queue &queue = system::default_queue())
{
    BOOST_STATIC_ASSERT(is_device_iterator<InputIterator1>::value);
    BOOST_STATIC_ASSERT(is_device_iterator<InputIterator2>::value);
    BOOST_STATIC_ASSERT(is_device_iterator<OutputIterator>::value);
    return detail::transform_if_impl(
        first, last, stencil, result, function, predicate, false, queue
    );
}

} // end compute namespace
} // end boost namespace
