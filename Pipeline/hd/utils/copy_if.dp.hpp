#pragma once

#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/scatter_if.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/for_each.hpp>
#include <boost/compute/iterator/permutation_iterator.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>
#include <boost/compute/system.hpp>

using boost::compute::command_queue;

template <typename UnaryFunction, typename Predicate, typename Tuple>
struct invoked_unary_transform_if_with_stencil_functor {
    UnaryFunction unary_op;
    Predicate pred;
    Tuple t;

    invoked_unary_transform_if_with_stencil_functor(UnaryFunction unary_op_, Predicate pred_, Tuple t_)
        : unary_op(unary_op_), pred(pred_), t(t_) {}
}; // end unary_transform_if_with_stencil_functor

namespace boost {
namespace compute {
namespace detail {

template <typename UnaryFunction, typename Predicate, typename Tuple>
inline meta_kernel &operator<<(meta_kernel &k, const invoked_unary_transform_if_with_stencil_functor<UnaryFunction, Predicate, Tuple> &func) {
    k << k.if_(func.pred(get<1>()(func.t))) << "\n"
      << get<2>()(func.t) << "=" << func.unary_op(get<0>()(func.t)) << ";\n";
    return k;
}

} // namespace detail
} // namespace compute
} // namespace boost

// reference: thrust/detail/internal_functional.h
template <typename UnaryFunction, typename Predicate>
struct unary_transform_if_with_stencil_functor {
    UnaryFunction unary_op;
    Predicate pred;

    unary_transform_if_with_stencil_functor(UnaryFunction unary_op_, Predicate pred_)
        : unary_op(unary_op_), pred(pred_) {}

    template <typename Tuple>
    inline invoked_unary_transform_if_with_stencil_functor<UnaryFunction, Predicate, Tuple> operator()(Tuple t) {
        return invoked_unary_transform_if_with_stencil_functor(unary_op, pred, t);
    }
}; // end unary_transform_if_with_stencil_functor

// reference: https://github.com/NVIDIA/thrust/blob/fa54f2c6f1217237953f27ddf67f901b6b34fbdd/thrust/system/detail/generic/transform_if.inl
template <typename InputIterator1,
          typename InputIterator2,
          typename ForwardIterator,
          typename UnaryFunction,
          typename Predicate>
void transform_if(InputIterator1 first,
                             InputIterator1 last,
                             InputIterator2 stencil,
                             ForwardIterator result,
                             UnaryFunction unary_op,
                             Predicate pred) {
    typedef unary_transform_if_with_stencil_functor<UnaryFunction, Predicate> UnaryTransformIfFunctor;

    // make an iterator tuple
    typedef boost::tuple<InputIterator1, InputIterator2, ForwardIterator> IteratorTuple;
    typedef boost::compute::zip_iterator<IteratorTuple> ZipIterator;
    typedef typename std::iterator_traits<InputIterator1>::difference_type IndexType1;
    typedef typename std::iterator_traits<InputIterator2>::difference_type IndexType2;
    typedef typename std::iterator_traits<ForwardIterator>::difference_type IndexType3;

    size_t n = std::distance(first, last);
    boost::compute::for_each(boost::compute::make_zip_iterator(boost::make_tuple(first, stencil, result)),
                             boost::compute::make_zip_iterator(boost::make_tuple(last, stencil, result)), // TODO
                             UnaryTransformIfFunctor(unary_op, pred));
    // ZipIterator zipped_result = boost::compute::make_zip_iterator(boost::make_tuple(first, stencil, result)) + (last - first);

    // return boost::get<2>(zipped_result.get_iterator_tuple());
} // end transform_if()

// reference: https://github.com/NVIDIA/thrust/blob/fa54f2c6f1217237953f27ddf67f901b6b34fbdd/thrust/system/detail/generic/scatter.inl
template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator,
          typename Predicate>
void scatter_if(InputIterator1 first,
                InputIterator1 last,
                InputIterator2 map,
                InputIterator3 stencil,
                RandomAccessIterator output,
                Predicate pred) {
    typedef typename std::iterator_traits<InputIterator1>::value_type InputType;
    ::transform_if(first, last, stencil, boost::compute::make_permutation_iterator(output, map), boost::compute::identity<InputType>(), pred);
} // end scatter_if()

// see boost::compute::copy_if and thrust::system::detail::generic::detail::copy_if
// reference: https://github.com/NVIDIA/thrust/blob/fa54f2c6f1217237953f27ddf67f901b6b34fbdd/thrust/system/detail/generic/copy_if.inl
template <typename IndexType = std::size_t, class InputIterator1, class InputIterator2, class OutputIterator, class Predicate>
inline OutputIterator copy_if(InputIterator1 first,
                              InputIterator1 last,
                              InputIterator2 stencil,
                              OutputIterator result,
                              Predicate predicate,
                              command_queue &queue = boost::compute::system::default_queue()) {
    using namespace boost::compute;
    BOOST_STATIC_ASSERT(is_device_iterator<InputIterator1>::value);
    BOOST_STATIC_ASSERT(is_device_iterator<InputIterator2>::value);
    BOOST_STATIC_ASSERT(is_device_iterator<OutputIterator>::value);
    typedef typename std::iterator_traits<InputIterator1>::value_type T;

    IndexType n = std::distance(first, last);

    // compute {0,1} predicates
    boost::compute::vector<IndexType> predicates(n);
    /*
     * According to OpenCL 1.2 Specification Section 6.1.1,
     * The value true expands to the integer constant 1 and
     * the value false expands to the integer constant 0.
     * So predicate_to_integral may not be needed.
     */
    boost::compute::transform(stencil,
                              stencil + n,
                              predicates.begin(),
                              predicate); // thrust::predicate_to_integral(pred));
    boost::compute::system::default_queue().finish();

    // scan {0,1} predicates
    boost::compute::vector<IndexType> scatter_indices(n);
    boost::compute::exclusive_scan(predicates.begin(),
                                   predicates.end(),
                                   scatter_indices.begin(),
                                   static_cast<IndexType>(0),
                                   boost::compute::plus<IndexType>());
    boost::compute::system::default_queue().finish();

    // scatter the true elements
    ::scatter_if(first,
                 last,
                 scatter_indices.begin(),
                 predicates.begin(),
                 result,
                 boost::compute::identity<IndexType>());
    boost::compute::system::default_queue().finish();

    // find the end of the new sequence
    IndexType output_size = scatter_indices[n - 1] + predicates[n - 1];

    return result + output_size;
}