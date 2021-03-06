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

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <hd/permutation_iterator.hpp>

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
    typedef typename std::iterator_traits<Iterator>::difference_type
        difference_type;
    struct stride_functor {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename boost::iterators::counting_iterator<difference_type> CountingIterator;
    typedef typename boost::iterators::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename heimdall::util::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

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
