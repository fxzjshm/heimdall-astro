/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/find_giants.h"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "hd/utils.hpp"

// TESTING only
#include "hd/stopwatch.h"

#include <iostream>
#include <boost/tuple/tuple.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <sycl/algorithm/copy.hpp>
#include <sycl/algorithm/copy_if.hpp>
#include <sycl/algorithm/adjacent_difference.hpp>
#include <sycl/algorithm/reduce_by_key.hpp>
#include <ZipIterator.hpp>
//#define PRINT_BENCHMARKS


template <typename T> struct greater_than_val {
  T val;
  greater_than_val(T val_) : val(val_) {}
  inline bool operator()(T x) const { return x > val; }
};

template <typename T>
struct maximum_first {
  inline T operator()(T a, T b) const {
    if constexpr (::is_tuple<T>::value) {
      return std::get<0>(a) >= std::get<0>(b) ? a : b;
    } else {
      return a >= b ? a : b;
    }
  }
};

template <typename T> struct nearby {
  // binary_operator removed in c++17
  using first_argument_type = T;
  using second_argument_type = T;
  T max_dist;
  nearby(T max_dist_) : max_dist(max_dist_) {}
  inline bool operator()(T a, T b) const { return b <= a + max_dist; }
};
template <typename T> struct not_nearby {
  T max_dist;
  not_nearby(T max_dist_) : max_dist(max_dist_) {}
  inline bool operator()(T b, T a) const { return b > a + max_dist; }
};

template <typename T> struct plus_one {
  inline T operator()(T x) const { return x + 1; }
};

class GiantFinder_impl {
  device_vector_wrapper<hd_float> d_giant_data;
  device_vector_wrapper<hd_size> d_giant_data_inds;
  device_vector_wrapper<hd_size> d_giant_data_segments;
  device_vector_wrapper<hd_size> d_giant_data_seg_ids;

public:
  hd_error exec(const hd_float *d_data, hd_size count, hd_float thresh,
                hd_size merge_dist,
                device_vector_wrapper<hd_float> &d_giant_peaks,
                device_vector_wrapper<hd_size> &d_giant_inds,
                device_vector_wrapper<hd_size> &d_giant_begins,
                device_vector_wrapper<hd_size> &d_giant_ends) {
    // This algorithm works by extracting all samples in the time series
    //   above thresh (the giant_data), segmenting those samples into
    //   isolated giants (based on merge_dist), and then computing the
    //   details of each giant into the d_giant_* arrays using
    //   reduce_by_key and some scatter operations.

    typedef heimdall::util::device_pointer<const hd_float> const_float_ptr;
    // typedef thrust::system::cuda::pointer<const hd_float> const_float_ptr;
    typedef heimdall::util::device_pointer<hd_float> float_ptr;
    typedef heimdall::util::device_pointer<hd_size> size_ptr;

    /*const_*/float_ptr d_data_begin(const_cast<hd_float*>(d_data));
    /*const_*/float_ptr d_data_end(const_cast<hd_float*>(d_data) + count);

#ifdef PRINT_BENCHMARKS
    Stopwatch timer;

    timer.start();
#endif

    // Note: The calls to Thrust in this function are retagged to use a
    //         custom temporary memory allocator (cached_allocator.cuh).
    //       This turns out to be critical to performance!

    // Quickly count how much giant data there is so we know the space needed
    /* DPCT_ORIG     hd_size giant_data_count =
     * thrust::count_if(thrust::retag<my_tag>(d_data_begin),*/
    
    // TODO: check this reduce
    hd_size giant_data_count = sycl::impl::count_if(execution_policy,
        d_data_begin, d_data_end, greater_than_val<hd_float>(thresh), std::plus());
    // std::cout << "GIANT_DATA_COUNT = " << giant_data_count << std::endl;
    // We can bail early if there are no giants at all
    if (0 == giant_data_count) {
      // std::cout << "**** Found ZERO giants" << std::endl;
      return HD_NO_ERROR;
    }

#ifdef PRINT_BENCHMARKS
    dpct::get_default_queue().wait();
    timer.stop();
    std::cout << "count_if time:           " << timer.getTime() << " s"
              << std::endl;
    timer.reset();

    timer.start();
#endif

    d_giant_data.resize(giant_data_count);
    d_giant_data_inds.resize(giant_data_count);

#ifdef PRINT_BENCHMARKS
    dpct::get_default_queue().wait();
    timer.stop();
    std::cout << "giant_data resize time:  " << timer.getTime() << " s"
              << std::endl;
    timer.reset();

    // Copy all of the giant data and their locations into one place

    timer.start();
#endif
    /*
    hd_size giant_data_count2 = 
      copy_if(make_zip_iterator(make_tuple(thrust::retag<my_tag>(d_data_begin),
                                           make_counting_iterator(0u))),
              make_zip_iterator(make_tuple(thrust::retag<my_tag>(d_data_begin),
                                           make_counting_iterator(0u)))+count,
              (d_data_begin), // the stencil
              make_zip_iterator(make_tuple(thrust::retag<my_tag>(d_giant_data.begin()),
                                           thrust::retag<my_tag>(d_giant_data_inds.begin()))),
              greater_than_val<hd_float>(thresh))
      - make_zip_iterator(make_tuple(thrust::retag<my_tag>(d_giant_data.begin()),
                                     thrust::retag<my_tag>(d_giant_data_inds.begin())));
    */
    // tried serveral zip_iterators but not compatible with this algorithm implmention
    // TODO: check this copy_if
    hd_size giant_data_count2 =
        sycl::impl::copy_if(execution_policy,
            boost::iterators::make_counting_iterator(0u),
            boost::iterators::make_counting_iterator(0u) + count,
            (d_data_begin), // the stencil
            d_giant_data_inds.begin(),
            greater_than_val<hd_float>(thresh))
        - d_giant_data_inds.begin();
    sycl::impl::copy(execution_policy,
            boost::make_permutation_iterator(d_data_begin, d_giant_data_inds.begin()),
            boost::make_permutation_iterator(d_data_begin, d_giant_data_inds.begin()) + giant_data_count2,
            d_giant_data.begin()
    );

    /* """
    no known conversion from 'ZipIter<dpct::device_pointer<float>,
    boost::iterators::counting_iterator<unsigned int, boost::use_default, boost::use_default>>::reference'
    (aka 'ZipRef<float, const unsigned int>') to 'std::tuple<float, unsigned int>'
       """ */
    /*
    hd_size giant_data_count2 = 
      sycl::impl::copy_if(execution_policy,
              ZipIter(d_data_begin, boost::iterators::make_counting_iterator(0u)),
              ZipIter(d_data_begin, boost::iterators::make_counting_iterator(0u))+count,
              (d_data_begin), // the stencil
              ZipIter(d_giant_data.begin(), d_giant_data_inds.begin()),
              greater_than_val<hd_float>(thresh))
      - ZipIter(d_giant_data.begin(), d_giant_data_inds.begin());
    */

#ifdef PRINT_BENCHMARKS
    dpct::get_default_queue().wait();
    timer.stop();
    std::cout << "giant_data copy_if time: " << timer.getTime() << " s"
              << std::endl;
    timer.reset();

    timer.start();
#endif

    // Create an array of head flags indicating candidate segments
    // thrust::device_vector<int> d_giant_data_segments(giant_data_count);
    d_giant_data_segments.resize(giant_data_count);
    sycl::impl::adjacent_difference(
        execution_policy,
        d_giant_data_inds.begin(),
        d_giant_data_inds.end(),
        d_giant_data_segments.begin(), not_nearby<hd_size>(merge_dist));

    // hd_size giant_count_quick = thrust::count(d_giant_data_segments.begin(),
    //                                          d_giant_data_segments.end(),
    //                                          (int)true);

    // The first element is implicitly a segment head
    if (giant_data_count > 0) {
      d_giant_data_segments.front() = 0;
      // d_giant_data_segments.front() = 1;
    }

    // thrust::device_vector<hd_size>
    // d_giant_data_seg_ids(d_giant_data_segments.size());
    d_giant_data_seg_ids.resize(d_giant_data_segments.size());

    sycl::impl::inclusive_scan(
        execution_policy,
        d_giant_data_segments.begin(),
        d_giant_data_segments.end(),
        d_giant_data_seg_ids.begin(),
        static_cast<hd_size>(0), std::plus());

    // We extract the number of giants from the end of the exclusive scan
    // hd_size giant_count = d_giant_data_seg_ids.back() +
    //  d_giant_data_segments.back() + 1;
    hd_size giant_count = d_giant_data_seg_ids.back() + 1;
    // hd_size giant_count = d_giant_data_seg_ids.back() +
    //  d_giant_data_segments.back();

    // Report back the actual number of giants found
    // total_giant_count = giant_count;

#ifdef PRINT_BENCHMARKS
    dpct::get_default_queue().wait();
    timer.stop();
    std::cout << "giant segments time:     " << timer.getTime() << " s"
              << std::endl;
    timer.reset();

    timer.start();
#endif

    hd_size new_giants_offset = d_giant_peaks.size();
    // Allocate space for the new giants
    d_giant_peaks.resize(d_giant_peaks.size() + giant_count);
    d_giant_inds.resize(d_giant_inds.size() + giant_count);
    d_giant_begins.resize(d_giant_begins.size() + giant_count);
    d_giant_ends.resize(d_giant_ends.size() + giant_count);
    float_ptr new_giant_peaks_begin(&d_giant_peaks[new_giants_offset]);
    size_ptr new_giant_inds_begin(&d_giant_inds[new_giants_offset]);
    size_ptr new_giant_begins_begin(&d_giant_begins[new_giants_offset]);
    size_ptr new_giant_ends_begin(&d_giant_ends[new_giants_offset]);

#ifdef PRINT_BENCHMARKS
    dpct::get_default_queue().wait();
    timer.stop();
    std::cout << "giants resize time:      " << timer.getTime() << " s"
              << std::endl;
    timer.reset();

    timer.start();
#endif

    // Now we find the value (snr) and location (time) of each giant's maximum
    /*
    hd_size giant_count2 = 
      reduce_by_key(thrust::retag<my_tag>(d_giant_data_inds.begin()), // the keys
                    thrust::retag<my_tag>(d_giant_data_inds.end()),
                    make_zip_iterator(make_tuple(thrust::retag<my_tag>(d_giant_data.begin()),
                                                 thrust::retag<my_tag>(d_giant_data_inds.begin()))),
                    thrust::make_discard_iterator(), // the keys output
                    make_zip_iterator(make_tuple(thrust::retag<my_tag>(new_giant_peaks_begin),
                                                 thrust::retag<my_tag>(new_giant_inds_begin))),
                    nearby<hd_size>(merge_dist),
                    maximum_first<thrust::tuple<hd_float,hd_size> >())
      .second - make_zip_iterator(make_tuple(thrust::retag<my_tag>(new_giant_peaks_begin),
                                             thrust::retag<my_tag>(new_giant_inds_begin)));
    */
    hd_size giant_count2 =
        sycl::impl::reduce_by_key(
            execution_policy,
            d_giant_data_inds.begin(), // the keys
            d_giant_data_inds.end(),
            d_giant_data.begin(),
            new_giant_inds_begin, // the keys output
            new_giant_peaks_begin,
            nearby<hd_size>(merge_dist),
            maximum_first<hd_float>())
            .second -
        new_giant_peaks_begin;

#ifdef PRINT_BENCHMARKS
    dpct::get_default_queue().wait();
    timer.stop();
    std::cout << "reduce_by_key time:      " << timer.getTime() << " s"
              << std::endl;
    timer.reset();

    timer.start();
#endif

    // Now we make the first segment explicit
    if (giant_count > 0) {
      d_giant_data_segments[0] = 1;
    }

    // Create arrays of the beginning and end indices of each giant
    sycl::impl::scatter_if(execution_policy,
               d_giant_data_inds.begin(), d_giant_data_inds.end(),
               d_giant_data_seg_ids.begin(), d_giant_data_segments.begin(),
               new_giant_begins_begin);
    sycl::impl::scatter_if(execution_policy,
        boost::iterators::make_transform_iterator(d_giant_data_inds.begin(), plus_one<hd_size>()),
        boost::iterators::make_transform_iterator(d_giant_data_inds.end() - 1, plus_one<hd_size>()),
        d_giant_data_seg_ids.begin(), d_giant_data_segments.begin() + 1,
        new_giant_ends_begin);

    if (giant_count > 0) {
      d_giant_ends.back() = d_giant_data_inds.back() + 1;
    }

#ifdef PRINT_BENCHMARKS
    dpct::get_default_queue().wait();
    timer.stop();
    std::cout << "begin/end copy_if time:  " << timer.getTime() << " s"
              << std::endl;
    timer.reset();

    std::cout << "--------------------" << std::endl;
#endif

    return HD_NO_ERROR;
  }
};

// Public interface (wrapper for implementation)
GiantFinder::GiantFinder()
  : m_impl(new GiantFinder_impl) {}
hd_error GiantFinder::exec(const hd_float *d_data, hd_size count,
                           hd_float thresh, hd_size merge_dist,
                           device_vector_wrapper<hd_float> &d_giant_peaks,
                           device_vector_wrapper<hd_size> &d_giant_inds,
                           device_vector_wrapper<hd_size> &d_giant_begins,
                           device_vector_wrapper<hd_size> &d_giant_ends) {
  return m_impl->exec(d_data, count, thresh, merge_dist, d_giant_peaks,
                      d_giant_inds, d_giant_begins, d_giant_ends);
}
