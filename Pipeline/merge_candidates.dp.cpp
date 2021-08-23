/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/merge_candidates.h"
#include "hd/utils.hpp"
#include <boost/compute.hpp>

typedef std::tuple<hd_float, hd_size, hd_size, hd_size, hd_size, hd_size, hd_size> candidate_tuple;

struct merge_candidates_functor {
    /* DPCT_ORIG 	inline __host__ __device__*/
    inline candidate_tuple operator()(const candidate_tuple &c1,
                                      const candidate_tuple &c2) const {
        hd_float snr1 = std::get<0>(c1);
        hd_size ind1 = std::get<1>(c1);
        hd_size begin1 = std::get<2>(c1);
        hd_size end1 = std::get<3>(c1);
        hd_size filter_ind1 = std::get<4>(c1);
        hd_size dm_ind1 = std::get<5>(c1);
        hd_size members1 = std::get<6>(c1);

        hd_float snr2 = std::get<0>(c2);
        hd_size ind2 = std::get<1>(c2);
        hd_size begin2 = std::get<2>(c2);
        hd_size end2 = std::get<3>(c2);
        hd_size filter_ind2 = std::get<4>(c2);
        hd_size dm_ind2 = std::get<5>(c2);
        hd_size members2 = std::get<6>(c2);

        if (snr1 >= snr2) {
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
        } else {
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

hd_error merge_candidates(hd_size count,
                          hd_size *d_labels,
                          ConstRawCandidates d_cands,
                          RawCandidates d_groups) {
    typedef dpct::device_pointer<hd_float> float_iterator;
    typedef dpct::device_pointer<hd_size> size_iterator;
    typedef dpct::device_pointer<const hd_float> const_float_iterator;
    typedef dpct::device_pointer<const hd_size> const_size_iterator;

    size_iterator labels_begin(d_labels);

    float_iterator cand_peaks_begin(d_cands.peaks);
    size_iterator cand_inds_begin(d_cands.inds);
    size_iterator cand_begins_begin(d_cands.begins);
    size_iterator cand_ends_begin(d_cands.ends);
    size_iterator cand_filter_inds_begin(d_cands.filter_inds);
    size_iterator cand_dm_inds_begin(d_cands.dm_inds);
    size_iterator cand_members_begin(d_cands.members);

    float_iterator group_peaks_begin(d_groups.peaks);
    size_iterator group_inds_begin(d_groups.inds);
    size_iterator group_begins_begin(d_groups.begins);
    size_iterator group_ends_begin(d_groups.ends);
    size_iterator group_filter_inds_begin(d_groups.filter_inds);
    size_iterator group_dm_inds_begin(d_groups.dm_inds);
    size_iterator group_members_begin(d_groups.members);

    // Sort by labels and remember permutation
    boost::compute::vector<hd_size> d_permutation(count);
    boost::compute::iota(d_permutation.begin(), d_permutation.end(), 0);
    boost::compute::sort(labels_begin, labels_begin + count, d_permutation.begin());

    // Merge giants into groups according to the label
    boost::compute::reduce_by_key(
        labels_begin, labels_begin + count,
        boost::compute::make_permutation_iterator(
            boost::compute::make_zip_iterator(boost::make_tuple(cand_peaks_begin,
                                                                cand_inds_begin,
                                                                cand_begins_begin,
                                                                cand_ends_begin,
                                                                cand_filter_inds_begin,
                                                                cand_dm_inds_begin,
                                                                cand_members_begin)),
            d_permutation.begin()),
        boost::compute::discard_iterator(), // keys output
        boost::compute::make_zip_iterator(boost::make_tuple(group_peaks_begin,
                                                            group_inds_begin,
                                                            group_begins_begin,
                                                            group_ends_begin,
                                                            group_filter_inds_begin,
                                                            group_dm_inds_begin,
                                                            group_members_begin)),
        boost::compute::equal_to<hd_size>(), merge_candidates_functor());

    return HD_NO_ERROR;
}
