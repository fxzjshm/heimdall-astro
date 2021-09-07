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
#include "hd/utils/buffer_iterator.dp.hpp"
#include <boost/compute/system.hpp>

struct RemoveBaselinePlan_impl;

struct RemoveBaselinePlan {
	RemoveBaselinePlan();
	hd_error exec(boost::compute::buffer_iterator<hd_float> d_data,
	              hd_size   count,
	              hd_size   smooth_radius,
                  boost::compute::command_queue& queue = boost::compute::system::default_queue());
private:
	boost::shared_ptr<RemoveBaselinePlan_impl> m_impl;
};
