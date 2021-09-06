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
#include "hd/utils/buffer_iterator.dp.hpp"
#include "hd/utils/device_vector_wrapper.dp.hpp"

//#define PRINT_BENCHMARKS

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

#ifdef PRINT_BENCHMARKS
struct GiantFinder_profile {
    float count_if_time;
    float giant_data_resize_time;
    float giant_data_copy_if_time;
    float giant_segments_time;
    float giants_resize_time;
    float reduce_by_key_time;
    float begin_end_copy_if_time;
    float final_process_time;
};

extern GiantFinder_profile giant_finder_profile;
#endif // PRINT_BENCHMARKS
