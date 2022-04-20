/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/merge_candidates.h"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "hd/utils.hpp"

#include <sycl/algorithm/copy.hpp>
#include <sycl/algorithm/iota.hpp>
#include <sycl/algorithm/reduce_by_key.hpp>
#include <ZipIterator.hpp>

// modified for tuple incompatibility in algorithm implemention

typedef heimdall::util::device_pointer<hd_float> float_iterator;
typedef heimdall::util::device_pointer<hd_size> size_iterator;
typedef heimdall::util::device_pointer<const hd_float> const_float_iterator;
typedef heimdall::util::device_pointer<const hd_size> const_size_iterator;
typedef std::tuple<hd_float, hd_size, hd_size, hd_size, hd_size, hd_size, hd_size> candidate_tuple;

struct merge_candidates_functor {
    float_iterator cand_peaks_begin;
    // size_iterator  cand_inds_begin;
    // size_iterator  cand_begins_begin;
    // size_iterator  cand_ends_begin;
    // size_iterator  cand_filter_inds_begin;
    // size_iterator  cand_dm_inds_begin;
    // size_iterator  cand_members_begin;
    size_iterator  cand_members_out_begin;

    merge_candidates_functor(float_iterator cand_peaks_begin_,
                             // size_iterator  cand_inds_begin_,
                             // size_iterator  cand_begins_begin_,
                             // size_iterator  cand_ends_begin_,
                             // size_iterator  cand_filter_inds_begin_,
                             // size_iterator  cand_dm_inds_begin_,
                             // size_iterator  cand_members_begin_,
                             size_iterator  cand_members_out_begin_)
        : cand_peaks_begin(cand_peaks_begin_),
          // cand_inds_begin(cand_inds_begin_),
          // cand_begins_begin(cand_begins_begin_),
          // cand_ends_begin(cand_ends_begin_),
          // cand_filter_inds_begin(cand_filter_inds_begin_),
          // cand_dm_inds_begin(cand_dm_inds_begin_),
          // cand_members_begin(cand_members_begin_),
          cand_members_out_begin(cand_members_out_begin_) {}

    inline hd_size operator()(const hd_size &i1,
                              const hd_size &i2) const {

        hd_float snr1 = *(cand_peaks_begin + i1);
        // hd_size ind1 = *(cand_inds_begin + i1);
        // hd_size begin1 = *(cand_begins_begin + i1);
        // hd_size end1 = *(cand_ends_begin + i1);
        // hd_size filter_ind1 = *(cand_filter_inds_begin + i1);
        // hd_size dm_ind1 = *(cand_dm_inds_begin + i1);
        hd_size members1 = *(cand_members_out_begin + i1);

        hd_float snr2 = *(cand_peaks_begin + i2);
        // hd_size ind2 = *(cand_inds_begin + i2);
        // hd_size begin2 = *(cand_begins_begin + i2);
        // hd_size end2 = *(cand_ends_begin + i2);
        // hd_size filter_ind2 = *(cand_filter_inds_begin + i2);
        // hd_size dm_ind2 = *(cand_dm_inds_begin + i2);
        hd_size members2 = *(cand_members_out_begin + i2);

        if (snr1 >= snr2) {
            *(cand_members_out_begin + i1) = members1 + members2;
            return i1;
            // return std::make_tuple(
            //     snr1, ind1,
            //     //(begin1+begin2)/2,
            //     //(end1+end2)/2,
            //     // TODO: I think this is what gtools does
            //     // min((int)begin1, (int)begin2),
            //     // max((int)end1, (int)end2),
            //     // TODO: But this may be better
            //     begin1, end1, filter_ind1, dm_ind1,
            //     members1 + members2);
        } else {
            *(cand_members_out_begin + i2) = members1 + members2;
            return i2;
            // return std::make_tuple(snr2, ind2,
            //                        //(begin1+begin2)/2,
            //                        //(end1+end2)/2,
            //                        // min((int)begin1, (int)begin2),
            //                        // max((int)end1, (int)end2),
            //                        begin2, end2, filter_ind2,
            //                        dm_ind2, members1 + members2);
        }
    }
};

hd_error merge_candidates(hd_size count,
                          hd_size *d_labels,
                          ConstRawCandidates d_cands,
                          RawCandidates d_groups) {
    size_iterator labels_begin(d_labels);

    // TODO: fix for tuple not supporting non-const iterators
    RawCandidates d_cands_non_const = *(reinterpret_cast<RawCandidates *>(&d_cands));
    float_iterator cand_peaks_begin(d_cands_non_const.peaks);
    size_iterator cand_inds_begin(d_cands_non_const.inds);
    size_iterator cand_begins_begin(d_cands_non_const.begins);
    size_iterator cand_ends_begin(d_cands_non_const.ends);
    size_iterator cand_filter_inds_begin(d_cands_non_const.filter_inds);
    size_iterator cand_dm_inds_begin(d_cands_non_const.dm_inds);
    size_iterator cand_members_begin(d_cands_non_const.members);

    float_iterator group_peaks_begin(d_groups.peaks);
    size_iterator group_inds_begin(d_groups.inds);
    size_iterator group_begins_begin(d_groups.begins);
    size_iterator group_ends_begin(d_groups.ends);
    size_iterator group_filter_inds_begin(d_groups.filter_inds);
    size_iterator group_dm_inds_begin(d_groups.dm_inds);
    size_iterator group_members_begin(d_groups.members);

    // Sort by labels and remember permutation
    device_vector_wrapper<hd_size> d_permutation(count),
                                   d_permutation_out(count),
                                   d_labels_out(count),
                                   d_members_out(count);
    sycl::impl::iota(execution_policy,
        d_permutation.begin(), d_permutation.end());
    sycl::impl::sort_by_key(execution_policy,
        labels_begin, labels_begin + count, d_permutation.begin(),
        std::less());
    sycl::impl::copy(execution_policy, cand_members_begin, cand_members_begin + count, d_members_out.begin());

    // Merge giants into groups according to the label
    size_t group_count = 
        sycl::impl::reduce_by_key(execution_policy,
            labels_begin, labels_begin + count,
            d_permutation.begin(),
            d_labels_out.begin(), // discard_iterator(), // keys output
            d_permutation_out.begin(),
            std::equal_to<hd_size>(),
            merge_candidates_functor(
                cand_peaks_begin,
                // cand_inds_begin,
                // cand_begins_begin,
                // cand_ends_begin,
                // cand_filter_inds_begin,
                // cand_dm_inds_begin,
                // cand_members_begin,
                d_members_out.begin()
            )
        ).second - d_permutation_out.begin();
    execution_policy.get_queue().wait_and_throw();
    sycl::impl::copy(execution_policy,
        boost::iterators::make_permutation_iterator(cand_peaks_begin, d_permutation_out.begin()),
        boost::iterators::make_permutation_iterator(cand_peaks_begin, d_permutation_out.begin()) + group_count,
        group_peaks_begin
    );
    sycl::impl::copy(execution_policy,
        boost::iterators::make_permutation_iterator(cand_inds_begin, d_permutation_out.begin()),
        boost::iterators::make_permutation_iterator(cand_inds_begin, d_permutation_out.begin()) + group_count,
        group_inds_begin
    );
    sycl::impl::copy(execution_policy,
        boost::iterators::make_permutation_iterator(cand_begins_begin, d_permutation_out.begin()),
        boost::iterators::make_permutation_iterator(cand_begins_begin, d_permutation_out.begin()) + group_count,
        group_begins_begin
    );
    sycl::impl::copy(execution_policy,
        boost::iterators::make_permutation_iterator(cand_ends_begin, d_permutation_out.begin()),
        boost::iterators::make_permutation_iterator(cand_ends_begin, d_permutation_out.begin()) + group_count,
        group_ends_begin
    );
    sycl::impl::copy(execution_policy,
        boost::iterators::make_permutation_iterator(cand_filter_inds_begin, d_permutation_out.begin()),
        boost::iterators::make_permutation_iterator(cand_filter_inds_begin, d_permutation_out.begin()) + group_count,
        group_filter_inds_begin
    );
    sycl::impl::copy(execution_policy,
        boost::iterators::make_permutation_iterator(cand_dm_inds_begin, d_permutation_out.begin()),
        boost::iterators::make_permutation_iterator(cand_dm_inds_begin, d_permutation_out.begin()) + group_count,
        group_dm_inds_begin
    );
    sycl::impl::copy(execution_policy,
        boost::iterators::make_permutation_iterator(d_members_out.begin(), d_permutation_out.begin()),
        boost::iterators::make_permutation_iterator(d_members_out.begin(), d_permutation_out.begin()) + group_count,
        group_members_begin
    );
    execution_policy.get_queue().wait_and_throw();

    return HD_NO_ERROR;
}
