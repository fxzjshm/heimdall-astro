/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  This is taken from the Strided Range example supplied with Thrust.
 */

#pragma once

// see https://community.intel.com/t5/Intel-oneAPI-Threading-Building/tbb-task-has-not-been-declared/m-p/1255725#M14806
#if defined(_GLIBCXX_RELEASE) && 9 <=_GLIBCXX_RELEASE && _GLIBCXX_RELEASE <= 10
#define PSTL_USE_PARALLEL_POLICIES 0
#define _GLIBCXX_USE_TBB_PAR_BACKEND 0
#define _PSTL_PAR_BACKEND_SERIAL
#endif

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

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

    typedef typename std::iterator_traits<Iterator>::difference_type difference_type;

    struct stride_functor {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename oneapi::dpl::counting_iterator<difference_type>                   CountingIterator;
    typedef typename oneapi::dpl::transform_iterator<CountingIterator, stride_functor> TransformIterator;
    typedef typename oneapi::dpl::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

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
