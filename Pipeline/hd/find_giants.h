/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>

#include "hd/types.h"
#include "hd/error.h"
#include "hd/utils/wrappers.dp.hpp"

struct GiantFinder_impl;

struct GiantFinder {
	GiantFinder();
        hd_error exec(const boost::compute::buffer_iterator<hd_float> d_data, hd_size count, hd_float thresh,
                      hd_size merge_dist,
                      device_vector_wrapper<hd_float> &d_giant_peaks,
                      device_vector_wrapper<hd_size> &d_giant_inds,
                      device_vector_wrapper<hd_size> &d_giant_begins,
                      device_vector_wrapper<hd_size> &d_giant_ends);

private:
	boost::shared_ptr<GiantFinder_impl> m_impl;
};
