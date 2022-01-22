/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/find_giants.h"
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/dpl/iterator>
#include <dpct/dpl_utils.hpp>
#include "hd/utils.hpp"

// TESTING only
#include "hd/stopwatch.h"
#include <iostream>
//#define PRINT_BENCHMARKS


template<typename T>
struct greater_than_val {
  T val;
  greater_than_val(T val_) : val(val_) {}
  inline 
  bool operator()(T x) const {
    return x > val;
  }
};

template<typename T>
struct maximum_first {
  inline 
  T operator()(T a, T b) const {
    return std::get<0>(a) >= std::get<0>(b) ? a : b;
  }
};

template<typename T>
struct nearby {
  T max_dist;
  nearby(T max_dist_) : max_dist(max_dist_) {}
  inline 
  bool operator()(T a, T b) const {
    return b <= a + max_dist;
  }
};
template<typename T>
struct not_nearby {
  T max_dist;
  not_nearby(T max_dist_) : max_dist(max_dist_) {}
  inline 
  bool operator()(T b, T a) const {
    return b > a + max_dist;
  }
};

template<typename T>
struct plus_one {
  inline 
  T operator()(T x) const { return x+1; }
};

class GiantFinder_impl {
  device_vector_wrapper<hd_float> d_giant_data;
  device_vector_wrapper<hd_size>  d_giant_data_inds;
  device_vector_wrapper<int>      d_giant_data_segments;
  device_vector_wrapper<hd_size>  d_giant_data_seg_ids;
public:
  hd_error exec(const hd_float* d_data, hd_size count,
                hd_float thresh, hd_size merge_dist,
                device_vector_wrapper<hd_float>& d_giant_peaks,
                device_vector_wrapper<hd_size>&  d_giant_inds,
                device_vector_wrapper<hd_size>&  d_giant_begins,
                device_vector_wrapper<hd_size>&  d_giant_ends) {
    // This algorithm works by extracting all samples in the time series
    //   above thresh (the giant_data), segmenting those samples into
    //   isolated giants (based on merge_dist), and then computing the
    //   details of each giant into the d_giant_* arrays using
    //   reduce_by_key and some scatter operations.

    typedef dpct::device_pointer<const hd_float> const_float_ptr;
    //typedef thrust::system::cuda::pointer<const hd_float> const_float_ptr;
    typedef dpct::device_pointer<hd_float>             float_ptr;
    typedef dpct::device_pointer<hd_size>              size_ptr;
  
    const_float_ptr d_data_begin(d_data);
    const_float_ptr d_data_end(d_data + count);
  
#ifdef PRINT_BENCHMARKS
    Stopwatch timer;
  
    timer.start();
#endif
  
    // Note: The calls to Thrust in this function are retagged to use a
    //         custom temporary memory allocator (cached_allocator.cuh).
    //       This turns out to be critical to performance!
    
    // Quickly count how much giant data there is so we know the space needed
    hd_size giant_data_count = std::count_if(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        d_data_begin, d_data_end, greater_than_val<hd_float>(thresh));
    //std::cout << "GIANT_DATA_COUNT = " << giant_data_count << std::endl;
    // We can bail early if there are no giants at all
    if( 0 == giant_data_count ) {
      //std::cout << "**** Found ZERO giants" << std::endl;
      return HD_NO_ERROR;
    }
  
#ifdef PRINT_BENCHMARKS
    dpct::get_current_device().queues_wait_and_throw();
    timer.stop();
    std::cout << "count_if time:           " << timer.getTime() << " s" << std::endl;
    timer.reset();
  
    timer.start();
#endif
  
    d_giant_data.resize(giant_data_count);
    d_giant_data_inds.resize(giant_data_count);
  
#ifdef PRINT_BENCHMARKS
    dpct::get_current_device().queues_wait_and_throw();
    timer.stop();
    std::cout << "giant_data resize time:  " << timer.getTime() << " s" << std::endl;
    timer.reset();
  
    // Copy all of the giant data and their locations into one place
  
    timer.start();
#endif
  
    hd_size giant_data_count2 = 
      dpct::copy_if(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        oneapi::dpl::make_zip_iterator(
            d_data_begin, dpct::make_counting_iterator(0u)
        ),
        oneapi::dpl::make_zip_iterator(
           d_data_begin, dpct::make_counting_iterator(0u)
        )+count,
        (d_data_begin), // the stencil
        oneapi::dpl::make_zip_iterator(
           d_giant_data.begin(), d_giant_data_inds.begin()
        ),
        greater_than_val<hd_float>(thresh))
      - oneapi::dpl::make_zip_iterator(
          d_giant_data.begin(), d_giant_data_inds.begin()
        );
  
#ifdef PRINT_BENCHMARKS
    dpct::get_current_device().queues_wait_and_throw();
    timer.stop();
    std::cout << "giant_data copy_if time: " << timer.getTime() << " s" << std::endl;
    timer.reset();
  
    timer.start();
#endif
  
    // Create an array of head flags indicating candidate segments
    //thrust::device_vector<int> d_giant_data_segments(giant_data_count);
    d_giant_data_segments.resize(giant_data_count);
    oneapi::dpl::adjacent_difference(
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      d_giant_data_inds.begin(),
      d_giant_data_inds.end(),
      d_giant_data_segments.begin(), not_nearby<hd_size>(merge_dist));
  
    //hd_size giant_count_quick = thrust::count(d_giant_data_segments.begin(),
    //                                          d_giant_data_segments.end(),
    //                                          (int)true);
  
    // The first element is implicitly a segment head
    if( giant_data_count > 0 ) {
      d_giant_data_segments.front() = 0;
      //d_giant_data_segments.front() = 1;
    }
  
    //thrust::device_vector<hd_size> d_giant_data_seg_ids(d_giant_data_segments.size());
    d_giant_data_seg_ids.resize(d_giant_data_segments.size());
    
    oneapi::dpl::inclusive_scan(
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      d_giant_data_segments.begin(),
      d_giant_data_segments.end(),
      d_giant_data_seg_ids.begin());
  
    // We extract the number of giants from the end of the exclusive scan
    //hd_size giant_count = d_giant_data_seg_ids.back() +
    //  d_giant_data_segments.back() + 1;
    hd_size giant_count = d_giant_data_seg_ids.back() + 1;
    //hd_size giant_count = d_giant_data_seg_ids.back() +
    //  d_giant_data_segments.back();
  
    // Report back the actual number of giants found
    //total_giant_count = giant_count;
    
#ifdef PRINT_BENCHMARKS
    dpct::get_current_device().queues_wait_and_throw();
    timer.stop();
    std::cout << "giant segments time:     " << timer.getTime() << " s" << std::endl;
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
    size_ptr  new_giant_inds_begin(&d_giant_inds[new_giants_offset]);
    size_ptr  new_giant_begins_begin(&d_giant_begins[new_giants_offset]);
    size_ptr  new_giant_ends_begin(&d_giant_ends[new_giants_offset]);
  
#ifdef PRINT_BENCHMARKS
    dpct::get_current_device().queues_wait_and_throw();
    timer.stop();
    std::cout << "giants resize time:      " << timer.getTime() << " s" << std::endl;
    timer.reset();
  
    timer.start();
#endif
    
  
    // Now we find the value (snr) and location (time) of each giant's maximum
    hd_size giant_count2 = 
      oneapi::dpl::reduce_by_segment(
        oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
        d_giant_data_inds.begin(), // the keys
        d_giant_data_inds.end(),
        oneapi::dpl::make_zip_iterator(
          d_giant_data.begin(), d_giant_data_inds.begin()),
          oneapi::dpl::discard_iterator(),
          oneapi::dpl::make_zip_iterator(
            new_giant_peaks_begin, new_giant_inds_begin),
          nearby<hd_size>(merge_dist),
          maximum_first<std::tuple<hd_float,hd_size> >()
      ).second -
      oneapi::dpl::make_zip_iterator(
        new_giant_peaks_begin,
        new_giant_inds_begin
      );
  
#ifdef PRINT_BENCHMARKS
    dpct::get_current_device().queues_wait_and_throw();
    timer.stop();
    std::cout << "reduce_by_key time:      " << timer.getTime() << " s" << std::endl;
    timer.reset();
  
    timer.start();
#endif
  
    // Now we make the first segment explicit
    if( giant_count > 0 ) {
      d_giant_data_segments[0] = 1;
    }
  
    // Create arrays of the beginning and end indices of each giant
    /*
    DPCT1007:38: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
    */
    ::scatter_if(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                       d_giant_data_inds.begin(),
                       d_giant_data_inds.end(),
                       d_giant_data_seg_ids.begin(),
                       d_giant_data_segments.begin(),
                       new_giant_begins_begin);
    /*
    DPCT1007:39: Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.
    */
    ::scatter_if(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                       oneapi::dpl::make_transform_iterator(d_giant_data_inds.begin(), plus_one<hd_size>()),
                       oneapi::dpl::make_transform_iterator(d_giant_data_inds.end() - 1, plus_one<hd_size>()),
                       d_giant_data_seg_ids.begin(),
                       d_giant_data_segments.begin() + 1,
                       new_giant_ends_begin);
    
    if( giant_count > 0 ) {
      d_giant_ends.back() = d_giant_data_inds.back() + 1;
    }
  
  
#ifdef PRINT_BENCHMARKS
    dpct::get_current_device().queues_wait_and_throw();
    timer.stop();
    std::cout << "begin/end copy_if time:  " << timer.getTime() << " s" << std::endl;
    timer.reset();
  
    std::cout << "--------------------" << std::endl;
#endif
  
    return HD_NO_ERROR;
  }
  
};

// Public interface (wrapper for implementation)
GiantFinder::GiantFinder()
  : m_impl(new GiantFinder_impl) {}
hd_error GiantFinder::exec(const hd_float* d_data, hd_size count,
                           hd_float thresh, hd_size merge_dist,
                           device_vector_wrapper<hd_float>& d_giant_peaks,
                           device_vector_wrapper<hd_size>&  d_giant_inds,
                           device_vector_wrapper<hd_size>&  d_giant_begins,
                           device_vector_wrapper<hd_size>&  d_giant_ends) {
  return m_impl->exec(d_data, count,
                      thresh, merge_dist,
                      d_giant_peaks,
                      d_giant_inds,
                      d_giant_begins,
                      d_giant_ends);
}
