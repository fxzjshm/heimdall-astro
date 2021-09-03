/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include "hd/error.h"
#include "hd/types.h"

#include "hd/utils/buffer_iterator.dp.hpp"

hd_error measure_bandpass(const boost::compute::buffer_iterator<hd_byte> d_in,
                          hd_size        nsamps,
                          hd_size        nchans,
                          hd_size        nbits,
                          boost::compute::buffer_iterator<hd_float>      d_bandpass,
                          hd_float*      rms);

// Note: This returns an estimate from a sub-sample, not the exact average
hd_error measure_band_avg(const boost::compute::buffer_iterator<hd_byte> d_filterbank,
                          hd_size        nsamps,
                          hd_size        nchans,
                          hd_size        nbits,
                          boost::compute::buffer_iterator<hd_float>      d_band_avg);
/*
hd_error measure_band_outliers(const hd_byte* d_filterbank,
                               hd_size        nsamps,
                               hd_size        nchans,
                               hd_size        nbits,
                               hd_float       thresh,
                               hd_float*      d_band_outlier_fracs);
*/
