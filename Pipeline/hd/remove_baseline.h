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

struct RemoveBaselinePlan_impl;

struct RemoveBaselinePlan {
	RemoveBaselinePlan();
	hd_error exec(boost::compute::buffer_iterator<hd_float> d_data,
	              hd_size   count,
	              hd_size   smooth_radius);
private:
	boost::shared_ptr<RemoveBaselinePlan_impl> m_impl;
};
