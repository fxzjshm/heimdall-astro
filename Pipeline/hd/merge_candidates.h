/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include "hd/types.h"
#include "hd/types_on_device.dp.hpp"
#include "hd/error.h"

hd_error merge_candidates(hd_size               count,
                          boost::compute::buffer_iterator<hd_size> d_labels,
                          RawCandidatesOnDevice d_cands,
                          RawCandidatesOnDevice d_groups);
