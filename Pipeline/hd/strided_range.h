/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  This is taken from the Strided Range example supplied with Thrust.
 */

/* DPCT_ORIG #include <thrust/iterator/counting_iterator.h>*/
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
/* DPCT_ORIG #include <thrust/iterator/transform_iterator.h>*/

/* DPCT_ORIG #include <thrust/iterator/permutation_iterator.h>*/

/* DPCT_ORIG #include <thrust/functional.h>*/

// this example illustrates how to make strided access to a range of values
// examples:
// strided_range([0, 1, 2, 3, 4, 5, 6], 1) -> [0, 1, 2, 3, 4, 5, 6]
// strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
// strided_range([0, 1, 2, 3, 4, 5, 6], 3) -> [0, 3, 6]
// ...

// Note: The length of this range is round_up(len(in_range) / stride)
template <typename Iterator>
class strided_range {
public:
/* DPCT_ORIG     typedef typename thrust::iterator_difference<Iterator>::type
 * difference_type;*/
    typedef typename std::iterator_traits<Iterator>::difference_type
        difference_type;

/* DPCT_ORIG     struct stride_functor : public
   thrust::unary_function<difference_type,difference_type>
    {*/
    /*
    DPCT1044:11: thrust::unary_function was removed because std::unary_function
    has been deprecated in C++11. You may need to remove references to typedefs
    from thrust::unary_function in the class definition.
    */
    struct stride_functor {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

/* DPCT_ORIG         __host__ __device__*/

        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

/* DPCT_ORIG     typedef typename thrust::counting_iterator<difference_type>
 * CountingIterator;*/
    typedef typename oneapi::dpl::counting_iterator<difference_type>
        CountingIterator;
/* DPCT_ORIG     typedef typename thrust::transform_iterator<stride_functor,
 * CountingIterator> TransformIterator;*/
    typedef typename oneapi::dpl::transform_iterator<CountingIterator,
                                                     stride_functor>
        TransformIterator;
/* DPCT_ORIG     typedef typename
 * thrust::permutation_iterator<Iterator,TransformIterator>
 * PermutationIterator;*/
    typedef
        typename oneapi::dpl::permutation_iterator<Iterator, TransformIterator>
            PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};
