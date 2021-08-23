/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <boost/compute.hpp>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "hd/types.h"
#include "hd/error.h"
#include "hd/utils.hpp"

struct GiantFinder_impl;

struct GiantFinder {
	GiantFinder();
        hd_error exec(const hd_float *d_data, hd_size count, hd_float thresh,
                      hd_size merge_dist,
                      boost::compute::vector<hd_float> &d_giant_peaks,
                      boost::compute::vector<hd_size> &d_giant_inds,
                      boost::compute::vector<hd_size> &d_giant_begins,
                      boost::compute::vector<hd_size> &d_giant_ends);

private:
	boost::shared_ptr<GiantFinder_impl> m_impl;
};
