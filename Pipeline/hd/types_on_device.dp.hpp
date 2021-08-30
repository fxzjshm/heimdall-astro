#pragma once

#include "hd/types.h"
#include <boost/compute/iterator/buffer_iterator.hpp>

using boost::compute::buffer_iterator;
// added for device computing
struct RawCandidatesOnDevice {
	buffer_iterator<hd_float> peaks;
	buffer_iterator<hd_size> inds;
	buffer_iterator<hd_size> begins;
	buffer_iterator<hd_size> ends;
	buffer_iterator<hd_size> filter_inds;
	buffer_iterator<hd_size> dm_inds;
	buffer_iterator<hd_size> members;
};
