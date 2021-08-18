/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

// TODO: Any way to avoid including this here?
/* DPCT_ORIG #include <thrust/device_vector.h>*/

#include <boost/shared_ptr.hpp>

#include "hd/types.h"
#include "hd/error.h"

struct GiantFinder_impl;

struct GiantFinder {
	GiantFinder();
        hd_error exec(const hd_float *d_data, hd_size count, hd_float thresh,
                      hd_size merge_dist,
                      /* DPCT_ORIG thrust::device_vector<hd_float>&
                         d_giant_peaks,*/
                      dpct::device_vector<hd_float> &d_giant_peaks,
                      /* DPCT_ORIG thrust::device_vector<hd_size>&
                         d_giant_inds,*/
                      dpct::device_vector<hd_size> &d_giant_inds,
                      /* DPCT_ORIG thrust::device_vector<hd_size>&
                         d_giant_begins,*/
                      dpct::device_vector<hd_size> &d_giant_begins,
                      /* DPCT_ORIG thrust::device_vector<hd_size>&
                         d_giant_ends);*/
                      dpct::device_vector<hd_size> &d_giant_ends);

private:
	boost::shared_ptr<GiantFinder_impl> m_impl;
};
