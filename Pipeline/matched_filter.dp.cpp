/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/matched_filter.h"
#include "hd/strided_range.h"
#include "hd/utils.dp.hpp"

#include <boost/compute/iterator/strided_iterator.hpp>

using boost::compute::buffer_iterator;

// TODO: Add error checking to the methods in here
template <typename T> class MatchedFilterPlan_impl {
  device_vector_wrapper<T> m_scanned;
  hd_size m_max_width;

public:
  hd_error prep(const buffer_iterator<T> d_in, hd_size count, hd_size max_width) {
    m_max_width = max_width;
    boost::compute::buffer_iterator<T> d_in_begin(d_in);
    boost::compute::buffer_iterator<T> d_in_end(d_in + count);

    // Note: One extra element so that we include the final value
    m_scanned.resize(count + 1, 0);

    boost::compute::inclusive_scan(d_in_begin, d_in_end, m_scanned.begin() + 1);
    return HD_NO_ERROR;
  }

  // Note: This writes div_round_up(count + 1 - max_width, tscrunch) values to
  // d_out with a relative starting offset of max_width/2
  // Note: This does not apply any normalisation to the output
  hd_error exec(buffer_iterator<T> d_out, hd_size filter_width, hd_size tscrunch = 1) {
    // TODO: Check that prep( ) has been called
    // TODO: Check that width <= m_max_width

    boost::compute::buffer_iterator<T> d_out_begin(d_out);

    hd_size offset = m_max_width / 2;
    hd_size ahead = (filter_width - 1) / 2 + 1; // Divide and round up
    hd_size behind = filter_width / 2;          // Divide and round down
    hd_size out_count = m_scanned.size() - m_max_width;

    hd_size stride = tscrunch;

    typedef typename boost::compute::vector<T>::iterator Iterator;

    // Striding through the scanned array has the same effect as tscrunching
    // TODO: Think about this carefully. Does it do exactly what we want?
    strided_range<Iterator> in_range1(
        m_scanned.begin() + offset + ahead,
        m_scanned.begin() + offset + ahead + out_count, stride);
    strided_range<Iterator> in_range2(
        m_scanned.begin() + offset - behind,
        m_scanned.begin() + offset - behind + out_count, stride);
    boost::compute::strided_iterator in_range1_begin(m_scanned.begin() + offset + ahead, stride);
    boost::compute::strided_iterator in_range1_end(m_scanned.begin() + offset + ahead + out_count, stride);
    boost::compute::strided_iterator in_range2_begin(m_scanned.begin() + offset - behind, stride);
    boost::compute::strided_iterator in_range2_end(m_scanned.begin() + offset - behind + out_count, stride);

    boost::compute::transform(
        // in_range1.begin(), in_range1.end(), in_range2.begin(), d_out_begin,
        in_range1_begin, in_range1_end, in_range2_begin, d_out_begin,
        boost::compute::minus<T>());

    return HD_NO_ERROR;
  }
};

// Public interface (wrapper for implementation)
template <typename T>
MatchedFilterPlan<T>::MatchedFilterPlan()
    : m_impl(new MatchedFilterPlan_impl<T>) {}
template <typename T>
hd_error MatchedFilterPlan<T>::prep(const boost::compute::buffer_iterator<T> d_in, hd_size count,
                                    hd_size max_width) {
  return m_impl->prep(d_in, count, max_width);
  // return (*this)->prep(d_in, count, max_width);
}
template <typename T>
hd_error MatchedFilterPlan<T>::exec(boost::compute::buffer_iterator<T> d_out, hd_size filter_width,
                                    hd_size tscrunch) {
  return m_impl->exec(d_out, filter_width, tscrunch);
}

// Explicit template instantiations for types used by other compilation units
template struct MatchedFilterPlan<hd_float>;
template struct MatchedFilterPlan<int>;
