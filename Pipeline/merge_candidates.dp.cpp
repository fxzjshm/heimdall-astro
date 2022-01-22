/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/merge_candidates.h"

// see https://community.intel.com/t5/Intel-oneAPI-Threading-Building/tbb-task-has-not-been-declared/m-p/1255725#M14806
#if defined(_GLIBCXX_RELEASE) && 9 <=_GLIBCXX_RELEASE && _GLIBCXX_RELEASE <= 10
#define PSTL_USE_PARALLEL_POLICIES 0
#define _GLIBCXX_USE_TBB_PAR_BACKEND 0
#define _PSTL_PAR_BACKEND_SERIAL
#endif

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include <dpct/dpl_utils.hpp>
#include "hd/utils.hpp"

typedef std::tuple<hd_float, hd_size, hd_size, hd_size, hd_size, hd_size, hd_size> candidate_tuple;

struct merge_candidates_functor {
	inline candidate_tuple operator()(const candidate_tuple& c1,
	                                 const candidate_tuple& c2) const {
		hd_float snr1 = std::get<0>(c1);
		hd_size  ind1 = std::get<1>(c1);
		hd_size  begin1 = std::get<2>(c1);
		hd_size  end1 = std::get<3>(c1);
		hd_size  filter_ind1 = std::get<4>(c1);
		hd_size  dm_ind1 = std::get<5>(c1);
		hd_size  members1 = std::get<6>(c1);
		
		hd_float snr2 = std::get<0>(c2);
		hd_size  ind2 = std::get<1>(c2);
		hd_size  begin2 = std::get<2>(c2);
		hd_size  end2 = std::get<3>(c2);
		hd_size  filter_ind2 = std::get<4>(c2);
		hd_size  dm_ind2 = std::get<5>(c2);
		hd_size  members2 = std::get<6>(c2);
		
		if( snr1 >= snr2 ) {
			return std::make_tuple(snr1,
			                          ind1,
			                          //(begin1+begin2)/2,
			                          //(end1+end2)/2,
			                          // TODO: I think this is what gtools does
			                          //min((int)begin1, (int)begin2),
			                          //max((int)end1, (int)end2),
			                          // TODO: But this may be better
			                          begin1,
			                          end1,
			                          filter_ind1,
			                          dm_ind1,
			                          members1+members2);
		}
		else {
			return std::make_tuple(snr2,
			                          ind2,
			                          //(begin1+begin2)/2,
			                          //(end1+end2)/2,
			                          //min((int)begin1, (int)begin2),
			                          //max((int)end1, (int)end2),
			                          begin2,
			                          end2,
			                          filter_ind2,
			                          dm_ind2,
			                          members1+members2);
		}
	}
};

hd_error merge_candidates(hd_size            count,
                          hd_size*           d_labels,
                          ConstRawCandidates d_cands,
                          RawCandidates      d_groups)
{
	typedef dpct::device_pointer<hd_float> float_iterator;
	typedef dpct::device_pointer<hd_size>  size_iterator;
	typedef dpct::device_pointer<const hd_float> const_float_iterator;
	typedef dpct::device_pointer<const hd_size>  const_size_iterator;
	
	size_iterator  labels_begin(d_labels);
	
	// TODO: fix for tuple not supporting non-const iterators
	RawCandidates d_cands_non_const = *((RawCandidates *)(&d_cands));
	float_iterator cand_peaks_begin(d_cands_non_const.peaks);
	size_iterator cand_inds_begin(d_cands_non_const.inds);
	size_iterator cand_begins_begin(d_cands_non_const.begins);
	size_iterator cand_ends_begin(d_cands_non_const.ends);
	size_iterator cand_filter_inds_begin(d_cands_non_const.filter_inds);
	size_iterator cand_dm_inds_begin(d_cands_non_const.dm_inds);
	size_iterator cand_members_begin(d_cands_non_const.members);
	
	float_iterator group_peaks_begin(d_groups.peaks);
	size_iterator  group_inds_begin(d_groups.inds);
	size_iterator  group_begins_begin(d_groups.begins);
	size_iterator  group_ends_begin(d_groups.ends);
	size_iterator  group_filter_inds_begin(d_groups.filter_inds);
	size_iterator  group_dm_inds_begin(d_groups.dm_inds);
	size_iterator  group_members_begin(d_groups.members);
	
	// Sort by labels and remember permutation
	device_vector_wrapper<hd_size> d_permutation(count);
	dpct::iota(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), d_permutation.begin(), d_permutation.end());
	dpct::sort(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), labels_begin, labels_begin + count,
	                    d_permutation.begin());
	
	// Merge giants into groups according to the label
	oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        labels_begin, labels_begin + count,
        oneapi::dpl::make_permutation_iterator(
            oneapi::dpl::make_zip_iterator(cand_peaks_begin,
                                           cand_inds_begin,
                                           cand_begins_begin,
                                           cand_ends_begin,
                                           cand_filter_inds_begin,
                                           cand_dm_inds_begin,
                                           cand_members_begin),
            d_permutation.begin()),
        oneapi::dpl::discard_iterator(), // keys output
        oneapi::dpl::make_zip_iterator(group_peaks_begin,
                                       group_inds_begin,
                                       group_begins_begin,
                                       group_ends_begin,
                                       group_filter_inds_begin,
                                       group_dm_inds_begin,
                                       group_members_begin),
        std::equal_to<hd_size>(), merge_candidates_functor());
	
	return HD_NO_ERROR;
}
