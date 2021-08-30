/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/merge_candidates.h"

#include "hd/utils/wrappers.dp.hpp"
#include "hd/utils/reduce_by_key.dp.hpp"
#include <boost/compute/algorithm.hpp>
#include <boost/compute/iterator.hpp>

typedef boost::tuple<hd_float, hd_size, hd_size, hd_size, hd_size, hd_size, hd_size> candidate_tuple;

struct merge_candidates_functor {
    inline auto operator()() const {
    BOOST_COMPUTE_FUNCTION(candidate_tuple, merge_candidates_functor, (const candidate_tuple &c1, const candidate_tuple &c2), {
        hd_float snr1 = boost_tuple_get(c1, 0);
        hd_size ind1 = boost_tuple_get(c1, 1);
        hd_size begin1 = boost_tuple_get(c1, 2);
        hd_size end1 = boost_tuple_get(c1, 3);
        hd_size filter_ind1 = boost_tuple_get(c1, 4);
        hd_size dm_ind1 = boost_tuple_get(c1, 5);
        hd_size members1 = boost_tuple_get(c1, 6);

        hd_float snr2 = boost_tuple_get(c2, 0);
        hd_size ind2 = boost_tuple_get(c2, 1);
        hd_size begin2 = boost_tuple_get(c2, 2);
        hd_size end2 = boost_tuple_get(c2, 3);
        hd_size filter_ind2 = boost_tuple_get(c2, 4);
        hd_size dm_ind2 = boost_tuple_get(c2, 5);
        hd_size members2 = boost_tuple_get(c2, 6);

        if (snr1 >= snr2) {
            return {
                snr1, ind1,
                //(begin1+begin2)/2,
                //(end1+end2)/2,
                // TODO: I think this is what gtools does
                // min((int)begin1, (int)begin2),
                // max((int)end1, (int)end2),
                // TODO: But this may be better
                begin1, end1, filter_ind1, dm_ind1,
                members1 + members2
            };
        } else {
            return {
                snr2, ind2,
                //(begin1+begin2)/2,
                //(end1+end2)/2,
                // min((int)begin1, (int)begin2),
                // max((int)end1, (int)end2),
                begin2, end2, filter_ind2,
                dm_ind2, members1 + members2
            };
        }
    });
    return merge_candidates_functor;
    }
};

hd_error merge_candidates(hd_size count,
                          boost::compute::buffer_iterator<hd_size> d_labels,
                          RawCandidatesOnDevice d_cands,
                          RawCandidatesOnDevice d_groups) {
    typedef boost::compute::buffer_iterator<hd_float> float_iterator;
    typedef boost::compute::buffer_iterator<hd_size> size_iterator;
    typedef boost::compute::buffer_iterator<const hd_float> const_float_iterator;
    typedef boost::compute::buffer_iterator<const hd_size> const_size_iterator;

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
    device_vector_wrapper<hd_size> d_permutation(count);
    boost::compute::iota(d_permutation.begin(), d_permutation.end(), 0);
    boost::compute::sort_by_key(labels_begin, labels_begin + count, d_permutation.begin());

    // Merge giants into groups according to the label
    boost::compute::detail::dispatch_reduce_by_key_no_count_check(
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
        boost::compute::equal_to<hd_size>(), merge_candidates_functor()(),
        boost::compute::system::default_queue());

    return HD_NO_ERROR;
}
