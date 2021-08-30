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
/*
hd_error label_candidate_clusters(hd_size        count,
                                  const hd_size* d_begins,
                                  const hd_size* d_ends,
                                  hd_size        time_count,
                                  //const hd_size* d_beams,
                                  //hd_size        beam_count,
                                  const hd_size* d_filter_inds,
                                  hd_size        filter_count,
                                  const hd_size* d_dm_inds,
                                  hd_size        dm_count,
                                  hd_size        merge_dist,
                                  hd_size*       d_new_labels);
*/
/*
hd_error label_candidate_clusters(hd_size        count,
                                  const hd_size* d_begins,
                                  const hd_size* d_ends,
                                  ////const hd_size* d_beams,
                                  const hd_size* d_filter_inds,
                                  const hd_size* d_dm_inds,
                                  hd_size        time_tol,
                                  hd_size        filter_tol,
                                  hd_size        dm_tol,
                                  hd_size*       d_labels,
                                  hd_size*       label_count);
*/
hd_error label_candidate_clusters(hd_size            count,
                                  RawCandidatesOnDevice d_cands,
                                  hd_size            time_tol,
                                  hd_size            filter_tol,
                                  hd_size            dm_tol,
                                  boost::compute::buffer_iterator<hd_size> d_labels,
                                  hd_size*           label_count);
