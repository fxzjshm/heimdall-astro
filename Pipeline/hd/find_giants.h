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

// see https://community.intel.com/t5/Intel-oneAPI-Threading-Building/tbb-task-has-not-been-declared/m-p/1255725#M14806
#if defined(_GLIBCXX_RELEASE) && 9 <=_GLIBCXX_RELEASE && _GLIBCXX_RELEASE <= 10
#define PSTL_USE_PARALLEL_POLICIES 0
#define _GLIBCXX_USE_TBB_PAR_BACKEND 0
#define _PSTL_PAR_BACKEND_SERIAL
#endif

// TODO: Any way to avoid including this here?
#include <dpct/dpl_utils.hpp>

#include <boost/shared_ptr.hpp>

#include "hd/types.h"
#include "hd/error.h"
#include "hd/utils.hpp"

struct GiantFinder_impl;

struct GiantFinder {
	GiantFinder();
        hd_error exec(const hd_float *d_data, hd_size count, hd_float thresh,
                      hd_size merge_dist,
                      device_vector_wrapper<hd_float> &d_giant_peaks,
                      device_vector_wrapper<hd_size> &d_giant_inds,
                      device_vector_wrapper<hd_size> &d_giant_begins,
                      device_vector_wrapper<hd_size> &d_giant_ends);

private:
	boost::shared_ptr<GiantFinder_impl> m_impl;
};
