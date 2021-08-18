/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "hd/merge_candidates.h"

/* DPCT_ORIG #include <thrust/device_vector.h>*/
#include <dpct/dpl_utils.hpp>
/* DPCT_ORIG #include <thrust/sort.h>*/

/* DPCT_ORIG #include <thrust/functional.h>*/

/* DPCT_ORIG #include <thrust/count.h>*/

/* DPCT_ORIG #include <thrust/iterator/zip_iterator.h>*/

/* DPCT_ORIG #include <thrust/iterator/permutation_iterator.h>*/

/* DPCT_ORIG #include <thrust/iterator/discard_iterator.h>*/

/* DPCT_ORIG typedef thrust::tuple<hd_float,*/
typedef std::tuple<hd_float, hd_size, hd_size, hd_size, hd_size, hd_size,
                   hd_size>
    candidate_tuple;
/* DPCT_ORIG struct merge_candidates_functor : public
   thrust::binary_function<candidate_tuple, candidate_tuple, candidate_tuple>
   {*/
/*
DPCT1044:7: thrust::binary_function was removed because std::binary_function has
been deprecated in C++11. You may need to remove references to typedefs from
thrust::binary_function in the class definition.
*/
struct merge_candidates_functor {
/* DPCT_ORIG 	inline __host__ __device__*/
        inline candidate_tuple operator()(const candidate_tuple &c1,
                                          const candidate_tuple &c2) const {
/* DPCT_ORIG 		hd_float snr1 = thrust::get<0>(c1);*/
                hd_float snr1 = std::get<0>(c1);
/* DPCT_ORIG 		hd_size  ind1 = thrust::get<1>(c1);*/
                hd_size ind1 = std::get<1>(c1);
/* DPCT_ORIG 		hd_size  begin1 = thrust::get<2>(c1);*/
                hd_size begin1 = std::get<2>(c1);
/* DPCT_ORIG 		hd_size  end1 = thrust::get<3>(c1);*/
                hd_size end1 = std::get<3>(c1);
/* DPCT_ORIG 		hd_size  filter_ind1 = thrust::get<4>(c1);*/
                hd_size filter_ind1 = std::get<4>(c1);
/* DPCT_ORIG 		hd_size  dm_ind1 = thrust::get<5>(c1);*/
                hd_size dm_ind1 = std::get<5>(c1);
/* DPCT_ORIG 		hd_size  members1 = thrust::get<6>(c1);*/
                hd_size members1 = std::get<6>(c1);

/* DPCT_ORIG 		hd_float snr2 = thrust::get<0>(c2);*/
                hd_float snr2 = std::get<0>(c2);
/* DPCT_ORIG 		hd_size  ind2 = thrust::get<1>(c2);*/
                hd_size ind2 = std::get<1>(c2);
/* DPCT_ORIG 		hd_size  begin2 = thrust::get<2>(c2);*/
                hd_size begin2 = std::get<2>(c2);
/* DPCT_ORIG 		hd_size  end2 = thrust::get<3>(c2);*/
                hd_size end2 = std::get<3>(c2);
/* DPCT_ORIG 		hd_size  filter_ind2 = thrust::get<4>(c2);*/
                hd_size filter_ind2 = std::get<4>(c2);
/* DPCT_ORIG 		hd_size  dm_ind2 = thrust::get<5>(c2);*/
                hd_size dm_ind2 = std::get<5>(c2);
/* DPCT_ORIG 		hd_size  members2 = thrust::get<6>(c2);*/
                hd_size members2 = std::get<6>(c2);

                if( snr1 >= snr2 ) {
/* DPCT_ORIG 			return thrust::make_tuple(snr1,*/
                        return std::make_tuple(
                            snr1, ind1,
                            //(begin1+begin2)/2,
                            //(end1+end2)/2,
                            // TODO: I think this is what gtools does
                            // min((int)begin1, (int)begin2),
                            // max((int)end1, (int)end2),
                            // TODO: But this may be better
                            begin1, end1, filter_ind1, dm_ind1,
                            members1 + members2);
                }
		else {
/* DPCT_ORIG 			return thrust::make_tuple(snr2,*/
                        return std::make_tuple(snr2, ind2,
                                               //(begin1+begin2)/2,
                                               //(end1+end2)/2,
                                               // min((int)begin1, (int)begin2),
                                               // max((int)end1, (int)end2),
                                               begin2, end2, filter_ind2,
                                               dm_ind2, members1 + members2);
                }
	}
};

hd_error merge_candidates(hd_size            count,
                          hd_size*           d_labels,
                          ConstRawCandidates d_cands,
                          RawCandidates      d_groups)
{
/* DPCT_ORIG 	typedef thrust::device_ptr<hd_float> float_iterator;*/
        typedef dpct::device_pointer<hd_float> float_iterator;
/* DPCT_ORIG 	typedef thrust::device_ptr<hd_size>  size_iterator;*/
        typedef dpct::device_pointer<hd_size> size_iterator;
/* DPCT_ORIG 	typedef thrust::device_ptr<const hd_float>
 * const_float_iterator;*/
        typedef dpct::device_pointer<const hd_float> const_float_iterator;
/* DPCT_ORIG 	typedef thrust::device_ptr<const hd_size> const_size_iterator;*/
        typedef dpct::device_pointer<const hd_size> const_size_iterator;

        size_iterator  labels_begin(d_labels);
	
	const_float_iterator cand_peaks_begin(d_cands.peaks);
	const_size_iterator  cand_inds_begin(d_cands.inds);
	const_size_iterator  cand_begins_begin(d_cands.begins);
	const_size_iterator  cand_ends_begin(d_cands.ends);
	const_size_iterator  cand_filter_inds_begin(d_cands.filter_inds);
	const_size_iterator  cand_dm_inds_begin(d_cands.dm_inds);
	const_size_iterator  cand_members_begin(d_cands.members);
	
	float_iterator group_peaks_begin(d_groups.peaks);
	size_iterator  group_inds_begin(d_groups.inds);
	size_iterator  group_begins_begin(d_groups.begins);
	size_iterator  group_ends_begin(d_groups.ends);
	size_iterator  group_filter_inds_begin(d_groups.filter_inds);
	size_iterator  group_dm_inds_begin(d_groups.dm_inds);
	size_iterator  group_members_begin(d_groups.members);
	
	// Sort by labels and remember permutation
/* DPCT_ORIG 	thrust::device_vector<hd_size> d_permutation(count);*/
        dpct::device_vector<hd_size> d_permutation(count);
/* DPCT_ORIG 	thrust::sequence(d_permutation.begin(), d_permutation.end());*/
        dpct::iota(oneapi::dpl::execution::make_device_policy(
                       dpct::get_default_queue()),
                   d_permutation.begin(), d_permutation.end());
/* DPCT_ORIG 	thrust::sort_by_key(labels_begin, labels_begin + count,*/
        dpct::sort(oneapi::dpl::execution::make_device_policy(
                       dpct::get_default_queue()),
                   labels_begin, labels_begin + count, d_permutation.begin());

        // Merge giants into groups according to the label
	using thrust::reduce_by_key;
	using thrust::make_zip_iterator;
	using thrust::make_permutation_iterator;
/* DPCT_ORIG 	reduce_by_key(labels_begin, labels_begin + count,*/
        oneapi::dpl::reduce_by_segment(
            oneapi::dpl::execution::make_device_policy(
                dpct::get_default_queue()),
            labels_begin, labels_begin + count,
            /* DPCT_ORIG 	              make_permutation_iterator(*/
            oneapi::dpl::make_permutation_iterator(
                /* DPCT_ORIG
                   make_zip_iterator(thrust::make_tuple(cand_peaks_begin,
                                                                               cand_inds_begin,
                                                                               cand_begins_begin,
                                                                               cand_ends_begin,
                                                                               cand_filter_inds_begin,
                                                                               cand_dm_inds_begin,
                                                                               cand_members_begin)),*/
                oneapi::dpl::make_zip_iterator(cand_peaks_begin,
                                               cand_inds_begin),
                d_permutation.begin()),
            /* DPCT_ORIG 	              thrust::make_discard_iterator(),
             */
            dpct::discard_iterator(), // keys output
                                      /* DPCT_ORIG
                                         make_zip_iterator(thrust::make_tuple(group_peaks_begin,
                                                                                                 group_inds_begin,
                                                                                                 group_begins_begin,
                                                                                                 group_ends_begin,
                                                                                                 group_filter_inds_begin,
                                                                                                 group_dm_inds_begin,
                                                                                                 group_members_begin)),*/
            oneapi::dpl::make_zip_iterator(group_peaks_begin, group_inds_begin),
            thrust::equal_to<hd_size>(), merge_candidates_functor());

        return HD_NO_ERROR;
}
