/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include "hd/types.h"
#include "hd/error.h"

#include <boost/shared_ptr.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

struct GetRMSPlan_impl;

struct GetRMSPlan {
	GetRMSPlan();
	hd_float exec(boost::compute::buffer_iterator<hd_float> d_data, hd_size count);
private:
	boost::shared_ptr<GetRMSPlan_impl> m_impl;
};

// Convenience functions for one-off calls
hd_float get_rms(boost::compute::buffer_iterator<hd_float> d_data, hd_size count);
hd_error normalise(boost::compute::buffer_iterator<hd_float> d_data, hd_size count);
