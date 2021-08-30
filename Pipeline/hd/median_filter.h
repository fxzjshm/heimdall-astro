/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include "hd/error.h"
#include "hd/types.h"

#include <boost/compute/iterator.hpp>

using boost::compute::buffer_iterator;

hd_error median_filter3(const buffer_iterator<hd_float> d_in,
                        hd_size         count,
                        buffer_iterator<hd_float>       d_out);

hd_error median_filter5(const buffer_iterator<hd_float> d_in,
                        hd_size         count,
                        buffer_iterator<hd_float>       d_out);

hd_error median_scrunch3(const buffer_iterator<hd_float> d_in,
                         hd_size         count,
                         buffer_iterator<hd_float>       d_out);

hd_error median_scrunch5(const buffer_iterator<hd_float> d_in,
                         hd_size         count,
                         buffer_iterator<hd_float>       d_out);

// Note: This can operate 'in-place'
hd_error mean_filter2(const buffer_iterator<hd_float> d_in,
                      hd_size         count,
                      buffer_iterator<hd_float>       d_out);

hd_error linear_stretch(const buffer_iterator<hd_float> d_in,
                        hd_size         in_count,
                        buffer_iterator<hd_float>       d_out,
                        hd_size         out_count);

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 3
hd_error median_scrunch3_array(const buffer_iterator<hd_float> d_in,
                               hd_size         array_size,
                               hd_size         count,
                               buffer_iterator<hd_float>       d_out);

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 5
hd_error median_scrunch5_array(const buffer_iterator<hd_float> d_in,
                               hd_size         array_size,
                               hd_size         count,
                               buffer_iterator<hd_float>       d_out);

// Mean-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 2
hd_error mean_scrunch2_array(const buffer_iterator<hd_float> d_in,
                             hd_size         array_size,
                             hd_size         count,
                             buffer_iterator<hd_float>       d_out);
