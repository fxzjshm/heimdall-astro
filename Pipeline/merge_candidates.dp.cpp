/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/merge_candidates.h"

#include "hd/utils/func_with_source_string.dp.hpp"
#include "hd/utils/reduce_by_key.dp.hpp"
#include "hd/utils/type_defines.dp.hpp"
#include "hd/utils/wrappers.dp.hpp"
#include <boost/compute/algorithm.hpp>
#include <boost/compute/iterator.hpp>

//#include "hd/write_time_series.h"

typedef boost::tuple<hd_float, hd_size, hd_size, hd_size, hd_size, hd_size, hd_size> candidate_tuple;

const std::string common_source = type_define_source() + "typedef " + boost::compute::type_name<candidate_tuple>() + " TUPLE_TYPENAME;";

struct merge_candidates_functor {
    inline auto operator()() const {
    std::string source = ("{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
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
            return (TUPLE_TYPENAME){
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
            return (TUPLE_TYPENAME){
                snr2, ind2,
                //(begin1+begin2)/2,
                //(end1+end2)/2,
                // min((int)begin1, (int)begin2),
                // max((int)end1, (int)end2),
                begin2, end2, filter_ind2,
                dm_ind2, members1 + members2
            };
        }
    }));
    auto func = BOOST_COMPUTE_FUNCTION_WITH_NAME_AND_SOURCE_STRING(candidate_tuple, "merge_candidates_functor", (const candidate_tuple c1, const candidate_tuple c2), source.c_str());
    return func;
    }
};

template <typename T>
struct extract_struct_functor {
    // argument_wrapper cannot be used here as here needs a literal
    hd_size index;

    extract_struct_functor(hd_size index_) : index(index_) {}

    inline auto operator()() const {
        std::string type_name = boost::compute::type_name<T>();
        std::string name = std::string("extract_struct_functor_") + type_name + std::to_string(index); 
        std::string source = ("{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
            return boost_tuple_get(c, index_placeholder);
        }));
        boost::replace_first(source, "index_placeholder", std::to_string(index));
        auto func = BOOST_COMPUTE_FUNCTION_WITH_NAME_AND_SOURCE_STRING(T, name.c_str(), (const candidate_tuple c), source.c_str());
        return func;
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
    boost::compute::system::default_queue().finish();
    //write_device_time_series(labels_begin, count, 1.f, "merge_candidates.d_labels.1.not_tim");
    boost::compute::sort_by_key(labels_begin, labels_begin + count, d_permutation.begin());
    //write_device_time_series(labels_begin, count, 1.f, "merge_candidates.d_labels.2.not_tim");
    //write_vector(d_permutation, "merge_candidates.d_permutation.2");
    boost::compute::system::default_queue().finish();

    // Merge giants into groups according to the label
    // WARNING: BinaryFunction and BinaryPredicate is swapped for different API between thrust and Boost.Compute
    boost::compute::vector<candidate_tuple> d_out(count);
    hd_size group_count = 
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
            discard_iterator_wrapper(), // keys output
            d_out.begin(),
            merge_candidates_functor()(), boost::compute::equal_to<hd_size>(),
            boost::compute::system::default_queue()).second - d_out.begin();
    boost::compute::system::default_queue().finish();
    boost::compute::transform(d_out.begin(), d_out.begin() + group_count, group_peaks_begin, extract_struct_functor<hd_float>(0)());
    boost::compute::transform(d_out.begin(), d_out.begin() + group_count, group_inds_begin, extract_struct_functor<hd_size>(1)());
    boost::compute::transform(d_out.begin(), d_out.begin() + group_count, group_begins_begin, extract_struct_functor<hd_size>(2)());
    boost::compute::transform(d_out.begin(), d_out.begin() + group_count, group_ends_begin, extract_struct_functor<hd_size>(3)());
    boost::compute::transform(d_out.begin(), d_out.begin() + group_count, group_filter_inds_begin, extract_struct_functor<hd_size>(4)());
    boost::compute::transform(d_out.begin(), d_out.begin() + group_count, group_dm_inds_begin, extract_struct_functor<hd_size>(5)());
    boost::compute::transform(d_out.begin(), d_out.begin() + group_count, group_members_begin, extract_struct_functor<hd_size>(6)());
    boost::compute::system::default_queue().finish();
    //write_device_time_series(d_key_out_1.begin(), d_key_out_count_1, 1.f, "merge_candidates.d_key_out_1.not_tim");
    //write_device_time_series(d_key_out_2.begin(), d_key_out_count_2, 1.f, "merge_candidates.d_key_out_2.not_tim");
    //write_device_time_series(d_groups.peaks, d_key_out_count_1, 1.f, "merge_candidates.group_peaks_begin.not_tim");
    //write_device_time_series(d_groups.members, d_key_out_count_2, 1.f, "merge_candidates.group_members_begin.not_tim");

    return HD_NO_ERROR;
}
