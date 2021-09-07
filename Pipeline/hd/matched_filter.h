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

template<typename T>
struct MatchedFilterPlan_impl;

template<typename T>
struct MatchedFilterPlan {
  typedef T value_type;

  MatchedFilterPlan();
  hd_error prep(const boost::compute::buffer_iterator<T> d_in, hd_size count, hd_size max_width, boost::compute::command_queue& queue = boost::compute::system::default_queue());
  // Note: This writes (count + 1 - max_width) values to d_out
  //         with a relative starting offset of max_width/2
  // Note: This does not apply any normalisation to the output
  hd_error exec(boost::compute::buffer_iterator<T> d_out, hd_size width, hd_size tscrunch=1, boost::compute::command_queue& queue = boost::compute::system::default_queue());

private:
  boost::shared_ptr<MatchedFilterPlan_impl<T> > m_impl;
};
