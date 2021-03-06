/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <sycl/execution_policy>
#include <sycl/algorithm/copy.hpp>
#include <sycl/algorithm/reduce.hpp>
#include <sycl/algorithm/gather.hpp>
#include <sycl/helpers/sycl_usm_vector.hpp>
#include <dpct/dpl_extras/iterators.h>

#include <vector>
#include <memory>
#include <numeric>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <sstream>
#include <iomanip>
#include <string>
#include <fstream>

#include "hd/pipeline.h"
#include "hd/maths.h"
#include "hd/clean_filterbank_rfi.h"

#include "hd/remove_baseline.h"
#include "hd/matched_filter.h"
#include "hd/get_rms.h"
#include "hd/find_giants.h"
#include "hd/label_candidate_clusters.h"
#include "hd/merge_candidates.h"

#include "hd/DataSource.h"
#include "hd/ClientSocket.h"
#include "hd/SocketException.h"
#include "hd/stopwatch.h"         // For benchmarking
//#include "hd/write_time_series.h" // For debugging
#include "hd/utils.hpp"
#include "hd/ThreadPool.h"

#include <dedisp.h>

//#define HD_BENCHMARK

#ifdef HD_BENCHMARK
  void start_timer(Stopwatch& timer) { timer.start(); }
  void stop_timer(Stopwatch &timer) {
   dpct::get_current_device().queues_wait_and_throw(); timer.stop();
  }
#else
  void start_timer(Stopwatch& timer) { }
  void stop_timer(Stopwatch& timer) { }
#endif // HD_BENCHMARK

#include <utility>
#include <cmath>

sycl::sycl_execution_policy<> execution_policy;

// for host-vector and device_vector
template<typename T> using host_vector = sycl::helpers::usm_vector<T>;
template<typename T> using device_vector = device_vector_wrapper<T>;

 // For std::pair
template<typename T, typename U>
std::pair<T&,U&> tie(T& a, U& b) { return std::pair<T&,U&>(a,b); }

struct hd_pipeline_t {
  hd_params   params;
  dedisp_plan dedispersion_plan;
  //MPI_Comm    communicator;

  // Memory buffers used during pipeline execution
  std::vector<hd_byte>    h_clean_filterbank;
  host_vector<hd_byte>    h_dm_series;
  // Should be one every thread, not global
  //device_vector<hd_float> d_time_series;
  //device_vector<hd_float> d_filtered_series;
};

hd_error allocate_gpu(const hd_params params) {
  // TODO: This is just a simple proc-->GPU heuristic to get us started
  int gpu_count;
  gpu_count = dpct::dev_mgr::instance().device_count();
  //int proc_idx;
  //MPI_Comm comm = pl->communicator;
  //MPI_Comm_rank(comm, &proc_idx);
  int proc_idx = params.beam;
  int gpu_idx = params.gpu_id;
  
  try {
    dpct::dev_mgr::instance().select_device(gpu_idx);
    if( params.verbosity >= 1 ) {
      cout << "using device " << dpct::dev_mgr::instance().current_device().get_info<cl::sycl::info::device::name>() << endl;
    }
    dedisp_set_device(gpu_idx);
  } catch(sycl::exception e) {
    cerr << "Could not set device id to " << gpu_idx << ": "
         << e.what() << endl;
    return HD_INVALID_DEVICE_INDEX;
  }

  sycl::helpers::set_default_device(dpct::dev_mgr::instance().current_device());
  execution_policy = sycl::helpers::default_execution_policy();
  
  if( params.verbosity >= 1 ) {
    cout << "Process " << proc_idx << " using GPU " << gpu_idx << endl;
  }
#if defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
  int cerror;
  if( !params.yield_cpu ) {
    if( params.verbosity >= 2 ) {
      cout << "\tProcess " << proc_idx << " setting CPU to spin" << endl;
    }
    cerror = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    if( cerror != cudaSuccess ) {
      return throw_cuda_error(cerror);
    }
  }
  else {
    if( params.verbosity >= 2 ) {
      cout << "\tProcess " << proc_idx << " setting CPU to yield" << endl;
    }
    // Note: This Yield flag doesn't seem to work properly.
    //   The BlockingSync flag does the job, although it may interfere
    //     with GPU/CPU overlapping (not currently used).
    //cerror = cudaSetDeviceFlags(cudaDeviceScheduleYield);
    cerror = cudaSetDeviceFlags(cudaDeviceBlockingSync);
    if( cerror != cudaSuccess ) {
      return throw_cuda_error(cerror);
    }
  }
#else
  cerr << "yield_cpu is not supported yet in this implemention" << endl;
#endif // defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
  
  return HD_NO_ERROR;
}

unsigned int get_filter_index(unsigned int filter_width) {
  // This function finds log2 of the 32-bit power-of-two number v
  unsigned int v = filter_width;
  static const unsigned int b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 
                                   0xFF00FF00, 0xFFFF0000};
  /* register */ unsigned int r = (v & b[0]) != 0;
  for( int i=4; i>0; --i) {
    r |= ((v & b[i]) != 0) << i;
  }
  return r;
}

hd_error hd_create_pipeline(hd_pipeline* pipeline_, hd_params params) {
  // In sycl we should set GPU before creating device memory, otherwise the runtime often crash
  if( params.verbosity >= 2 ) {
    cout << "\tAllocating GPU..." << endl;
  }
  
  hd_error error = allocate_gpu(params);
  if( error != HD_NO_ERROR ) {
    return throw_error(error);
  }

  *pipeline_ = 0;
  
  // Note: We use a smart pointer here to automatically clean up after errors
#if __cplusplus <= 199711L
  typedef std::auto_ptr<hd_pipeline_t> smart_pipeline_ptr;
#else
  typedef std::unique_ptr<hd_pipeline_t> smart_pipeline_ptr;
#endif
  smart_pipeline_ptr pipeline = smart_pipeline_ptr(new hd_pipeline_t());
  if( !pipeline.get() ) {
    return throw_error(HD_MEM_ALLOC_FAILED);
  }
  
  pipeline->params = params;
  
  if( params.verbosity >= 3 ) {
    cout << "nchans = " << params.nchans << endl;
    cout << "dt     = " << params.dt << endl;
    cout << "f0     = " << params.f0 << endl;
    cout << "df     = " << params.df << endl;
  }
  
  if( params.verbosity >= 2 ) {
    cout << "\tCreating dedispersion plan..." << endl;
  }
  
  dedisp_error derror;
  derror = dedisp_create_plan(&pipeline->dedispersion_plan,
                              params.nchans, params.dt,
                              params.f0, params.df);
  if( derror != DEDISP_NO_ERROR ) {
    return throw_dedisp_error(derror);
  }
  // TODO: Consider loading a pre-generated DM list instead for flexibility
  derror = dedisp_generate_dm_list(pipeline->dedispersion_plan,
                                   pipeline->params.dm_min,
                                   pipeline->params.dm_max,
                                   pipeline->params.dm_pulse_width,
                                   pipeline->params.dm_tol);
  if( derror != DEDISP_NO_ERROR ) {
    return throw_dedisp_error(derror);
  }
  
  if( pipeline->params.use_scrunching ) {
    derror = dedisp_enable_adaptive_dt(pipeline->dedispersion_plan,
                                       pipeline->params.dm_pulse_width,
                                       pipeline->params.scrunch_tol);
    if( derror != DEDISP_NO_ERROR ) {
      return throw_dedisp_error(derror);
    }
  }
  
  *pipeline_ = pipeline.release();
  
  if( params.verbosity >= 2 ) {
    cout << "\tInitialisation complete." << endl;
  }
  
  /*
  if( params.verbosity >= 1 ) {
    cout << "Using Thrust v"
         << THRUST_MAJOR_VERSION << "."
         << THRUST_MINOR_VERSION << "."
         << THRUST_SUBMINOR_VERSION << endl;
  }
  */

  return HD_NO_ERROR;
}

hd_error hd_execute(hd_pipeline pl,
                    const hd_byte* h_filterbank, hd_size nsamps, hd_size nbits,
                    hd_size first_idx, hd_size* nsamps_processed) {
  hd_error error = HD_NO_ERROR;
  
  Stopwatch total_timer;
  Stopwatch memory_timer;
  Stopwatch clean_timer;
  Stopwatch dedisp_timer;
  Stopwatch communicate_timer;
  Stopwatch copy_timer;
  Stopwatch baseline_timer;
  Stopwatch normalise_timer;
  Stopwatch filter_timer;
  Stopwatch coinc_timer;
  Stopwatch giants_timer;
  Stopwatch candidates_timer;
  
  start_timer(total_timer);

  execution_policy = sycl::sycl_execution_policy(dpct::get_default_queue());
  
  start_timer(clean_timer);
  // Note: Filterbank cleaning must be done out-of-place
  hd_size nbytes = nsamps * pl->params.nchans * nbits / 8;
  start_timer(memory_timer);
  pl->h_clean_filterbank.resize(nbytes);
  std::vector<int>          h_killmask(pl->params.nchans, 1);
  stop_timer(memory_timer);
  
  if( pl->params.verbosity >= 2 ) {
    cout << "\tCleaning 0-DM filterbank..." << endl;
  }
  
  // Start by cleaning up the filterbank based on the zero-DM time series
  hd_float cleaning_dm = 0.f;
  if( pl->params.verbosity >= 3 ) {
    /*
    cout << "\tWriting dirty filterbank to disk..." << endl;
    write_host_filterbank(&h_filterbank[0],
                          pl->params.nchans, nsamps, nbits,
                          pl->params.dt, pl->params.f0, pl->params.df,
                          "dirty_filterbank.fil");
    */
  }
  // Note: We only clean the narrowest zero-DM signals; otherwise we
  //         start removing real stuff from higher DMs.
  error = clean_filterbank_rfi(pl->dedispersion_plan,
                               &h_filterbank[0],
                               nsamps,
                               nbits,
                               &pl->h_clean_filterbank[0],
                               &h_killmask[0],
                               cleaning_dm,
                               pl->params.dt,
                               pl->params.baseline_length,
                               pl->params.rfi_tol,
                               pl->params.rfi_min_beams,
                               pl->params.rfi_broad,
                               pl->params.rfi_narrow,
                               1);//pl->params.boxcar_max);
  if( error != HD_NO_ERROR ) {
    return throw_error(error);
  }

  if( pl->params.verbosity >= 2 ) {
    cout << "Applying manual killmasks" << endl;
  }

  error = apply_manual_killmasks (pl->dedispersion_plan,
                                  &h_killmask[0], 
                                  pl->params.num_channel_zaps,
                                  pl->params.channel_zaps);
  if( error != HD_NO_ERROR ) {
    return throw_error(error);
  }

  
  hd_size good_chan_count = std::reduce(h_killmask.begin(), h_killmask.end());
  hd_size bad_chan_count = pl->params.nchans - good_chan_count;
  if( pl->params.verbosity >= 2 ) {
    cout << "Bad channel count = " << bad_chan_count << endl;
  }
  
  // TESTING
  //h_clean_filterbank.assign(h_filterbank, h_filterbank+nbytes);
  
  stop_timer(clean_timer);
  
  if( pl->params.verbosity >= 3 ) {
    /*
    cout << "\tWriting killmask to disk..." << endl;
    std::ofstream killfile("killmask.dat");
    for( size_t i=0; i<h_killmask.size(); ++i ) {
      killfile << h_killmask[i] << "\n";
    }
    killfile.close();
    
    cout << "\tWriting cleaned filterbank to disk..." << endl;
    write_host_filterbank(&pl->h_clean_filterbank[0],
                          pl->params.nchans, nsamps, nbits,
                          pl->params.dt, pl->params.f0, pl->params.df,
                          "clean_filterbank.fil");
    */
  }
  if( pl->params.verbosity >= 2 ) {
    cout << "\tGenerating DM list..." << endl;
  }
  
  if( pl->params.verbosity >= 3 ) {
    cout << "dm_min = " << pl->params.dm_min << endl;
    cout << "dm_max = " << pl->params.dm_max << endl;
    cout << "dm_tol = " << pl->params.dm_tol << endl;
    cout << "dm_pulse_width = " << pl->params.dm_pulse_width << endl;
    cout << "nchans = " << pl->params.nchans << endl;
    cout << "dt = " << pl->params.dt << endl;
    
    cout << "dedisp nchans = " << dedisp_get_channel_count(pl->dedispersion_plan) << endl;
    cout << "dedisp dt = " << dedisp_get_dt(pl->dedispersion_plan) << endl;
    cout << "dedisp f0 = " << dedisp_get_f0(pl->dedispersion_plan) << endl;
    cout << "dedisp df = " << dedisp_get_df(pl->dedispersion_plan) << endl;
  }
  
  hd_size      dm_count = dedisp_get_dm_count(pl->dedispersion_plan);
  const float* dm_list  = dedisp_get_dm_list(pl->dedispersion_plan);
  
  const dedisp_size* scrunch_factors =
    dedisp_get_dt_factors(pl->dedispersion_plan);
  if (pl->params.verbosity >= 3 ) 
  {
    cout << "DM List for " << pl->params.dm_min << " to " << pl->params.dm_max << endl;
    for( hd_size i=0; i<dm_count; ++i ) {
      cout << dm_list[i] << endl;
    }
  }  

  if( pl->params.verbosity >= 2 ) {
    cout << "Scrunch factors:" << endl;
    for( hd_size i=0; i<dm_count; ++i ) {
      cout << scrunch_factors[i] << " ";
    }
    cout << endl;
  }
  
  // Set channel killmask for dedispersion
  dedisp_set_killmask(pl->dedispersion_plan, &h_killmask[0]);
  if (dedisp_get_max_delay(pl->dedispersion_plan) > nsamps)
  {
    cerr << "maximum DM delay=" << dedisp_get_max_delay(pl->dedispersion_plan) << endl;
    cerr << "Number of samples=" << nsamps << endl;
    return throw_error(HD_TOO_FEW_NSAMPS);
  }
  
  hd_size nsamps_computed  = nsamps - dedisp_get_max_delay(pl->dedispersion_plan);
  hd_size series_stride    = nsamps_computed;
  
  // Report the number of samples that will be properly processed
  *nsamps_processed = nsamps_computed - pl->params.boxcar_max;
  
  if( pl->params.verbosity >= 3 ) {
    cout << "dm_count = " << dm_count << endl;
    cout << "max delay = " << dedisp_get_max_delay(pl->dedispersion_plan) << endl;
    cout << "nsamps_computed = " << nsamps_computed << endl;
  }
  
  hd_size beam = pl->params.beam;
  
  if( pl->params.verbosity >= 2 ) {
    cout << "\tAllocating memory for pipeline computations..." << endl;
  }
  
  start_timer(memory_timer);
  
  pl->h_dm_series.resize(series_stride * pl->params.dm_nbits/8 * dm_count);
  //pl->d_time_series.resize(series_stride);
  //pl->d_filtered_series.resize(series_stride, 0);
  
  stop_timer(memory_timer);
  
  RemoveBaselinePlan          baseline_remover;
  GetRMSPlan                  rms_getter;
  MatchedFilterPlan<hd_float> matched_filter_plan;
  GiantFinder                 giant_finder;

  device_vector_wrapper<hd_float> d_all_giant_peaks;
  device_vector_wrapper<hd_size> d_all_giant_inds;
  device_vector_wrapper<hd_size> d_all_giant_begins;
  device_vector_wrapper<hd_size> d_all_giant_ends;
  device_vector_wrapper<hd_size> d_all_giant_filter_inds;
  device_vector_wrapper<hd_size> d_all_giant_dm_inds;
  device_vector_wrapper<hd_size> d_all_giant_members;

  typedef heimdall::util::device_pointer<hd_float> dev_float_ptr;
  typedef heimdall::util::device_pointer<hd_size> dev_size_ptr;

  if( pl->params.verbosity >= 2 ) {
    cout << "\tDedispersing for DMs " << dm_list[0]
         << " to " << dm_list[dm_count-1] << "..." << endl;
  }
  
  // Dedisperse
  dedisp_error       derror;
  const dedisp_byte* in = &pl->h_clean_filterbank[0];
  dedisp_byte*       out = &pl->h_dm_series[0];
  dedisp_size        in_nbits = nbits;
  dedisp_size        in_stride = pl->params.nchans * in_nbits/8;
  dedisp_size        out_nbits = pl->params.dm_nbits;
  dedisp_size        out_stride = series_stride * out_nbits/8;
  unsigned           flags = 0;
  start_timer(dedisp_timer);
  derror = dedisp_execute_adv(pl->dedispersion_plan, nsamps,
                              in, in_nbits, in_stride,
                              out, out_nbits, out_stride,
                              flags);
  stop_timer(dedisp_timer);
  if( derror != DEDISP_NO_ERROR ) {
    return throw_dedisp_error(derror);
  }
  
  if( beam == 0 && first_idx == 0 ) {
    // TESTING
    //write_host_time_series((unsigned int*)out, nsamps_computed, out_nbits,
    //                       pl->params.dt, "dedispersed_0.tim");
  }
  
  if( pl->params.verbosity >= 2 ) {
    cout << "\tBeginning inner pipeline..." << endl;
  }
  
  // TESTING
  hd_size write_dm = 0;
  
  bool too_many_giants = false;
  
  {
  ThreadPool thread_pool(pl->params.ncpus);
  std::mutex m_mutex;
  // For each DM
  for( hd_size dm_idx=0; dm_idx<dm_count; ++dm_idx ) {
    auto inner_function = [dm_idx,
        &scrunch_factors, &nsamps_computed, &too_many_giants, &series_stride, &dm_list, &nsamps, &dm_count, &m_mutex, &pl,
        &d_all_giant_peaks, &d_all_giant_inds, &d_all_giant_begins, &d_all_giant_ends, &d_all_giant_filter_inds, &d_all_giant_dm_inds, &d_all_giant_members,
        &beam, &write_dm, &first_idx,
        &copy_timer, &baseline_timer, &normalise_timer, &filter_timer, &giants_timer]() -> hd_error {
    hd_error error = HD_NO_ERROR;
    thread_local RemoveBaselinePlan          baseline_remover;
    thread_local GetRMSPlan                  rms_getter;
    thread_local MatchedFilterPlan<hd_float> matched_filter_plan;
    thread_local GiantFinder                 giant_finder;
    thread_local device_vector_wrapper<hd_float> d_giant_peaks;
    thread_local device_vector_wrapper<hd_size>  d_giant_inds;
    thread_local device_vector_wrapper<hd_size>  d_giant_begins;
    thread_local device_vector_wrapper<hd_size>  d_giant_ends;
    thread_local device_vector_wrapper<hd_size>  d_giant_filter_inds;
    thread_local device_vector_wrapper<hd_size>  d_giant_dm_inds;
    thread_local device_vector_wrapper<hd_size>  d_giant_members;
    thread_local device_vector_wrapper<hd_float> d_time_series;
    thread_local device_vector_wrapper<hd_float> d_filtered_series;
    d_giant_peaks.clear();
    d_giant_inds.clear();
    d_giant_begins.clear();
    d_giant_ends.clear();
    d_giant_filter_inds.clear();
    d_giant_dm_inds.clear();
    d_giant_members.clear();
    d_time_series.resize(series_stride);
    d_filtered_series.resize(series_stride);
    //sycl::sycl_execution_policy<> local_execution_policy(sycl::queue(execution_policy.get_queue()));
    sycl::impl::fill(execution_policy, d_filtered_series.begin(), d_filtered_series.end(), 0);

    hd_size  cur_dm_scrunch = scrunch_factors[dm_idx];
    hd_size  cur_nsamps  = nsamps_computed / cur_dm_scrunch;
    hd_float cur_dt      = pl->params.dt * cur_dm_scrunch;
    
    // Bail if the candidate rate is too high
    if( too_many_giants ) {
      return HD_TOO_MANY_EVENTS;
    }
    
    if( pl->params.verbosity >= 4 ) {
      cout << "dm_idx     = " << dm_idx << endl;
      cout << "scrunch    = " << scrunch_factors[dm_idx] << endl;
      cout << "cur_nsamps = " << cur_nsamps << endl;
      cout << "dt0        = " << pl->params.dt << endl;
      cout << "cur_dt     = " << cur_dt << endl;
        
      cout << "\tBaselining and normalising each beam..." << endl;
    }

    hd_float *time_series = heimdall::util::get_raw_pointer(&d_time_series[0]);

    // Copy the time series to the device and convert to floats
    hd_size offset = dm_idx * series_stride * pl->params.dm_nbits/8;
    start_timer(copy_timer);
    execution_policy.get_queue().prefetch(&pl->h_dm_series[offset], cur_nsamps * pl->params.dm_nbits / 8).wait();
    switch( pl->params.dm_nbits ) {
    case 8:
      sycl::impl::copy(execution_policy,
                (unsigned char *)&pl->h_dm_series[offset],
                (unsigned char *)&pl->h_dm_series[offset] + cur_nsamps,
                d_time_series.begin());
      break;
    case 16:
      sycl::impl::copy(execution_policy,
                (unsigned short *)&pl->h_dm_series[offset],
                (unsigned short *)&pl->h_dm_series[offset] + cur_nsamps,
                d_time_series.begin());
      break;
    case 32:
      // Note: 32-bit implies float, not unsigned int
      sycl::impl::copy(execution_policy,
                (float *)&pl->h_dm_series[offset],
                (float *)&pl->h_dm_series[offset] + cur_nsamps,
                d_time_series.begin());
      break;
    default:
      return HD_INVALID_NBITS;
    }
    stop_timer(copy_timer);
    
    // Remove the baseline
    // -------------------
    // Note: Divided by 2 to form a smoothing radius
    hd_size nsamps_smooth = hd_size(pl->params.baseline_length /
                                    (2 * cur_dt));
    // Crop the smoothing length in case not enough samples
    start_timer(baseline_timer);
    
    // TESTING
    error = baseline_remover.exec(time_series, cur_nsamps, nsamps_smooth);
    stop_timer(baseline_timer);
    if( error != HD_NO_ERROR ) {
      return throw_error(error);
    }
    
    if( beam == 0 && dm_idx == write_dm && first_idx == 0 ) {
      // TESTING
      //write_device_time_series(time_series, cur_nsamps,
      //                         cur_dt, "baselined.tim");
    }
    // -------------------
    
    // Normalise
    // ---------
    start_timer(normalise_timer);
    hd_float rms = rms_getter.exec(time_series, cur_nsamps);
    sycl::impl::transform(
        execution_policy,
        d_time_series.begin(), d_time_series.end(),
        dpct::make_constant_iterator(hd_float(1.0) / rms),
        d_time_series.begin(),
        std::multiplies<hd_float>());
    stop_timer(normalise_timer);
    
    if( beam == 0 && dm_idx == write_dm && first_idx == 0 ) {
      // TESTING
      //write_device_time_series(time_series, cur_nsamps,
      //                         cur_dt, "normalised.tim");
    }
    // ---------
    
    // Prepare the boxcar filters
    // --------------------------
    // We can't process the first and last max-filter-width/2 samples
    hd_size rel_boxcar_max = pl->params.boxcar_max/cur_dm_scrunch;
    
    hd_size max_nsamps_filtered = cur_nsamps + 1 - rel_boxcar_max;
    // This is the relative offset into the time series of the filtered data
    hd_size cur_filtered_offset = rel_boxcar_max / 2;
    
    // Create and prepare matched filtering operations
    start_timer(filter_timer);
    // Note: Filter width is relative to the current time resolution
    matched_filter_plan.prep(time_series, cur_nsamps, rel_boxcar_max);
    stop_timer(filter_timer);
    // --------------------------

    hd_float *filtered_series = heimdall::util::get_raw_pointer(&d_filtered_series[0]);

    // Note: Filtering is done using a combination of tscrunching and
    //         'proper' boxcar convolution. The parameter min_tscrunch_width
    //         indicates how much of each to do. Raising min_tscrunch_width
    //         increases sensitivity but decreases performance and vice
    //         versa.
    
    // For each boxcar filter
    // Note: We cannot detect pulse widths < current time resolution
    for( hd_size filter_width=cur_dm_scrunch;
         filter_width<=pl->params.boxcar_max;
         filter_width*=2 ) {
      hd_size rel_filter_width = filter_width / cur_dm_scrunch;
      hd_size filter_idx = get_filter_index(filter_width);
      
      if( pl->params.verbosity >= 4 ) {
        cout << "Filtering each beam at width of " << filter_width << " filter_idx=" << filter_idx << endl;
      }
      
      // Note: Filter width is relative to the current time resolution
      hd_size rel_min_tscrunch_width =
          std::max(pl->params.min_tscrunch_width / cur_dm_scrunch, hd_size(1));
      hd_size rel_tscrunch_width =
          std::max(2 * rel_filter_width / rel_min_tscrunch_width, hd_size(1));
      // Filter width relative to cur_dm_scrunch AND tscrunch
      hd_size rel_rel_filter_width = rel_filter_width / rel_tscrunch_width;

      start_timer(filter_timer);
      
      error = matched_filter_plan.exec(filtered_series,
                                       rel_filter_width,
                                       rel_tscrunch_width);
      
      if( error != HD_NO_ERROR ) {
        return throw_error(error);
      }
      // Divide and round up
      hd_size cur_nsamps_filtered = ((max_nsamps_filtered-1)
                                     / rel_tscrunch_width + 1);
      hd_size cur_scrunch = cur_dm_scrunch * rel_tscrunch_width;
      
      if (pl->params.boxcar_renorm)
      {
        // recompute then RMS of the filtered time series, then use that for rescaling.
        // Note that this method reduces the S/N of injected pulses. For more information
        // see https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.2316G/abstract [Appendix A]
        hd_float rms = rms_getter.exec(filtered_series, cur_nsamps_filtered);
        sycl::impl::transform(
            execution_policy,
            heimdall::util::device_pointer<hd_float>(filtered_series),
            heimdall::util::device_pointer<hd_float>(filtered_series) +
                cur_nsamps_filtered,
            dpct::make_constant_iterator(hd_float(1.0) / rms),
            heimdall::util::device_pointer<hd_float>(filtered_series),
            std::multiplies<hd_float>());
      }
      else
      {
        // rescale the filtered time series (RMS ~ sqrt(time))
        dpct::constant_iterator<hd_float> norm_val_iter(1.0 / sqrt((hd_float)rel_filter_width));
        sycl::impl::transform(
            execution_policy,
            heimdall::util::device_pointer<hd_float>(filtered_series),
            heimdall::util::device_pointer<hd_float>(filtered_series) +
                cur_nsamps_filtered,
            norm_val_iter,
            heimdall::util::device_pointer<hd_float>(filtered_series),
            std::multiplies<hd_float>());
      }

      stop_timer(filter_timer);
      
      if( beam == 0 && dm_idx == write_dm && first_idx == 0 &&
          filter_width == 8 ) {
        // TESTING
        //write_device_time_series(filtered_series,
        //                         cur_nsamps_filtered,
        //                         cur_dt, "filtered.tim");
      }
      
      hd_size prev_giant_count = d_giant_peaks.size();
      
      if( pl->params.verbosity >= 4 ) {
        cout << "Finding giants..." << endl;
      }
      
      start_timer(giants_timer);

      if( pl->params.verbosity >= 4 ) {
        cerr << "pl->params.cand_sep_time=" << pl->params.cand_sep_time << " rel_rel_filter_width=" << rel_rel_filter_width << endl;
      }
      
      error = giant_finder.exec(filtered_series, cur_nsamps_filtered,
                                pl->params.detect_thresh,
                                //pl->params.cand_sep_time,
                                // Note: This was MB's recommendation
                                pl->params.cand_sep_time * rel_rel_filter_width,
                                d_giant_peaks,
                                d_giant_inds,
                                d_giant_begins,
                                d_giant_ends);
      
      if( error != HD_NO_ERROR ) {
        return throw_error(error);
      }

      // add this if to avoid crash (try to allocate 0-length buffer) when no giants found, and is also a minor optimize
      if(prev_giant_count < d_giant_peaks.size()){
      
        hd_size rel_cur_filtered_offset = (cur_filtered_offset /
                                           rel_tscrunch_width);

        sycl::impl::transform(
            execution_policy,
            d_giant_inds.begin() + prev_giant_count, d_giant_inds.end(),
            d_giant_inds.begin() + prev_giant_count,
            /*first_idx +*/ [=](auto _1) {
              return (_1 + rel_cur_filtered_offset) * cur_scrunch;
        });
        sycl::impl::transform(
            execution_policy,
            d_giant_begins.begin() + prev_giant_count, d_giant_begins.end(),
            d_giant_begins.begin() + prev_giant_count,
            /*first_idx +*/ [=](auto _1) {
              return (_1 + rel_cur_filtered_offset) * cur_scrunch;
        });
        sycl::impl::transform(
            execution_policy,
            d_giant_ends.begin() + prev_giant_count, d_giant_ends.end(),
            d_giant_ends.begin() + prev_giant_count,
            /*first_idx +*/ [=](auto _1) {
              return (_1 + rel_cur_filtered_offset) * cur_scrunch;
        });

        d_giant_filter_inds.resize(d_giant_peaks.size(), filter_idx);
        d_giant_dm_inds.resize(d_giant_peaks.size(), dm_idx);
        // Note: This could be used to track total member samples if desired
        d_giant_members.resize(d_giant_peaks.size(), 1);
      }
      
      stop_timer(giants_timer);
      
      // Bail if the candidate rate is too high
      hd_size total_giant_count = d_giant_peaks.size();
      hd_float data_length_mins = nsamps * pl->params.dt / 60.0;
      if ( pl->params.max_giant_rate && ( total_giant_count / data_length_mins > pl->params.max_giant_rate ) ) {
        too_many_giants = true;
        float searched = ((float) dm_idx * 100) / (float) dm_count;
        cout << "WARNING: exceeded max giants/min, DM [" << dm_list[dm_idx] << "] space searched " << searched << "%" << endl;
        break;
      }
      
    } // End of filter width loop
    // gather giant info
    {
      std::lock_guard lock(m_mutex);
      auto old_size = d_all_giant_peaks.size();
      auto new_size = old_size + d_giant_peaks.size();
      d_all_giant_peaks.resize(new_size);
      d_all_giant_inds.resize(new_size);
      d_all_giant_begins.resize(new_size);
      d_all_giant_ends.resize(new_size);
      d_all_giant_filter_inds.resize(new_size);
      d_all_giant_dm_inds.resize(new_size);
      d_all_giant_members.resize(new_size);
      sycl::impl::copy(execution_policy, d_giant_peaks.begin(), d_giant_peaks.end(), d_all_giant_peaks.begin() + old_size);
      sycl::impl::copy(execution_policy, d_giant_inds.begin(), d_giant_inds.end(), d_all_giant_inds.begin() + old_size);
      sycl::impl::copy(execution_policy, d_giant_begins.begin(), d_giant_begins.end(), d_all_giant_begins.begin() + old_size);
      sycl::impl::copy(execution_policy, d_giant_ends.begin(), d_giant_ends.end(), d_all_giant_ends.begin() + old_size);
      sycl::impl::copy(execution_policy, d_giant_filter_inds.begin(), d_giant_filter_inds.end(), d_all_giant_filter_inds.begin() + old_size);
      sycl::impl::copy(execution_policy, d_giant_dm_inds.begin(), d_giant_dm_inds.end(), d_all_giant_dm_inds.begin() + old_size);
      sycl::impl::copy(execution_policy, d_giant_members.begin(), d_giant_members.end(), d_all_giant_members.begin() + old_size);
      //execution_policy.get_queue().wait_and_throw();
    }
    return HD_NO_ERROR;
  };
  thread_pool.enqueue(inner_function);
  } // End of DM loop
  }

  hd_size giant_count = d_all_giant_peaks.size();
  if( pl->params.verbosity >= 2 ) {
    cout << "Giant count = " << giant_count << endl;
  }
  
  start_timer(candidates_timer);

  std::vector<hd_float> h_group_peaks;
  std::vector<hd_size> h_group_inds;
  std::vector<hd_size> h_group_begins;
  std::vector<hd_size> h_group_ends;
  std::vector<hd_size> h_group_filter_inds;
  std::vector<hd_size> h_group_dm_inds;
  std::vector<hd_size> h_group_members;
  std::vector<hd_float> h_group_dms;

  //if (!too_many_giants)
  //{
    device_vector_wrapper<hd_size> d_giant_labels(giant_count);
    hd_size *d_giant_labels_ptr = heimdall::util::get_raw_pointer(&d_giant_labels[0]);

    RawCandidates d_giants;
    d_giants.peaks = heimdall::util::get_raw_pointer(&d_all_giant_peaks[0]);
    d_giants.inds = heimdall::util::get_raw_pointer(&d_all_giant_inds[0]);
    d_giants.begins = heimdall::util::get_raw_pointer(&d_all_giant_begins[0]);
    d_giants.ends = heimdall::util::get_raw_pointer(&d_all_giant_ends[0]);
    d_giants.filter_inds = heimdall::util::get_raw_pointer(&d_all_giant_filter_inds[0]);
    d_giants.dm_inds = heimdall::util::get_raw_pointer(&d_all_giant_dm_inds[0]);
    d_giants.members = heimdall::util::get_raw_pointer(&d_all_giant_members[0]);

    hd_size filter_count = get_filter_index(pl->params.boxcar_max) + 1;

    if( pl->params.verbosity >= 2 ) {
      cout << "Grouping coincident candidates..." << endl;
    }

    ConstRawCandidates * const_d_giants = (ConstRawCandidates *) &d_giants;
  
    hd_size label_count;
    error = label_candidate_clusters(giant_count,
                                     *const_d_giants,
                                     pl->params.cand_sep_time,
                                     pl->params.cand_sep_filter,
                                     pl->params.cand_sep_dm,
                                     d_giant_labels_ptr,
                                     &label_count);
    if( error != HD_NO_ERROR ) {
      return throw_error(error);
    }
  
    hd_size group_count = label_count;
    if( pl->params.verbosity >= 2 ) {
      cout << "Candidate count = " << group_count << endl;
    }

    device_vector_wrapper<hd_float> d_group_peaks(group_count);
    device_vector_wrapper<hd_size> d_group_inds(group_count);
    device_vector_wrapper<hd_size> d_group_begins(group_count);
    device_vector_wrapper<hd_size> d_group_ends(group_count);
    device_vector_wrapper<hd_size> d_group_filter_inds(group_count);
    device_vector_wrapper<hd_size> d_group_dm_inds(group_count);
    device_vector_wrapper<hd_size> d_group_members(group_count);

    device_vector_wrapper<hd_float> d_group_dms(group_count);

    RawCandidates d_groups;
    d_groups.peaks = heimdall::util::get_raw_pointer(&d_group_peaks[0]);
    d_groups.inds = heimdall::util::get_raw_pointer(&d_group_inds[0]);
    d_groups.begins = heimdall::util::get_raw_pointer(&d_group_begins[0]);
    d_groups.ends = heimdall::util::get_raw_pointer(&d_group_ends[0]);
    d_groups.filter_inds = heimdall::util::get_raw_pointer(&d_group_filter_inds[0]);
    d_groups.dm_inds = heimdall::util::get_raw_pointer(&d_group_dm_inds[0]);
    d_groups.members = heimdall::util::get_raw_pointer(&d_group_members[0]);

    merge_candidates(giant_count,
                     d_giant_labels_ptr,
                     *const_d_giants,
                     d_groups);
  
    // Look up the actual DM of each group
    device_vector_wrapper<hd_float> d_dm_list(dm_list, dm_list + dm_count);

    sycl::impl::gather(execution_policy,
                d_group_dm_inds.begin(), d_group_dm_inds.end(),
                d_dm_list.begin(), d_group_dms.begin());

    // Device to host transfer of candidates
    // h_group_peaks = d_group_peaks;
    heimdall::util::copy(d_group_peaks, h_group_peaks);
    // h_group_inds = d_group_inds;
    heimdall::util::copy(d_group_inds, h_group_inds);
    // h_group_begins = d_group_begins;
    heimdall::util::copy(d_group_begins, h_group_begins);
    // h_group_ends = d_group_ends;
    heimdall::util::copy(d_group_ends, h_group_ends);
    // h_group_filter_inds = d_group_filter_inds;
    heimdall::util::copy(d_group_filter_inds, h_group_filter_inds);
    // h_group_dm_inds = d_group_dm_inds;
    heimdall::util::copy(d_group_dm_inds, h_group_dm_inds);
    // h_group_members = d_group_members;
    heimdall::util::copy(d_group_members, h_group_members);
    // h_group_dms = d_group_dms;
    heimdall::util::copy(d_group_dms, h_group_dms);
    //h_group_flags = d_group_flags;
    //heimdall::util::copy(d_group_flags, h_group_flags);
  //}
  
  if( pl->params.verbosity >= 2 ) {
    cout << "Writing output candidates, utc_start=" << pl->params.utc_start << endl;
  }

  char buffer[64];
  time_t now = pl->params.utc_start + (time_t) (first_idx / pl->params.spectra_per_second);
  strftime (buffer, 64, HD_TIMESTR, (struct tm*) gmtime(&now));

  std::stringstream ss;
  ss << std::setw(2) << std::setfill('0') << pl->params.beam+1;

  std::ostringstream oss;

  if ( pl->params.coincidencer_host != NULL && pl->params.coincidencer_port != -1 )
  {
    try 
    {
      ClientSocket client_socket ( pl->params.coincidencer_host, pl->params.coincidencer_port );

      strftime (buffer, 64, HD_TIMESTR, (struct tm*) gmtime(&(pl->params.utc_start)));

      oss <<  buffer << " ";

      time_t now = pl->params.utc_start + (time_t) (first_idx / pl->params.spectra_per_second);
      strftime (buffer, 64, HD_TIMESTR, (struct tm*) gmtime(&now));
      oss << buffer << " ";

      oss << first_idx << " ";
      oss << ss.str() << " ";
      oss << h_group_peaks.size() << endl;
      client_socket << oss.str();
      oss.flush();
      oss.str("");

      for (hd_size i=0; i<h_group_peaks.size(); ++i ) 
      {
        hd_size samp_idx = first_idx + h_group_inds[i];
        oss << h_group_peaks[i] << "\t"
                      << samp_idx << "\t"
                      << samp_idx * pl->params.dt << "\t"
                      << h_group_filter_inds[i] << "\t"
                      << h_group_dm_inds[i] << "\t"
                      << h_group_dms[i] << "\t"
                      << h_group_members[i] << "\t"
                      << first_idx + h_group_begins[i] << "\t"
                      << first_idx + h_group_ends[i] << endl;

        client_socket << oss.str();
        oss.flush();
        oss.str("");
      }
      // client_socket should close when it goes out of scope...
    }
    catch (SocketException& e )
    {
      std::cerr << "SocketException was caught:" << e.description() << "\n";
    }

  }
  else
  {
    if( pl->params.verbosity >= 2 )
      cout << "Output timestamp: " << buffer << endl;

    std::string filename = std::string(pl->params.output_dir) + "/" + std::string(buffer) + "_" + ss.str() + ".cand";

    if( pl->params.verbosity >= 2 )
      cout << "Output filename: " << filename << endl;

    std::ofstream cand_file(filename.c_str(), std::ios::out);
    if( pl->params.verbosity >= 2 )
      cout << "Dumping " << h_group_peaks.size() << " candidates to " << filename << endl;

    if (cand_file.good())
    {
      for( hd_size i=0; i<h_group_peaks.size(); ++i ) {
        hd_size samp_idx = first_idx + h_group_inds[i];
        cand_file << h_group_peaks[i] << "\t"
                  << samp_idx << "\t"
                  << samp_idx * pl->params.dt << "\t"
                  << h_group_filter_inds[i] << "\t"
                  << h_group_dm_inds[i] << "\t"
                  << h_group_dms[i] << "\t"
                  << h_group_members[i] << "\t"
                  << first_idx + h_group_begins[i] << "\t"
                  << first_idx + h_group_ends[i] << "\t"
                  << "\n";
      }
    }
    else
      cout << "Skipping dump due to bad file open on " << filename << endl;
    cand_file.close();
  }
    
  stop_timer(candidates_timer);
  
  stop_timer(total_timer);

#ifdef HD_BENCHMARK
  if( pl->params.verbosity >= 1 )
  {
  cout << "Mem alloc time:          " << memory_timer.getTime() << endl;
  cout << "0-DM cleaning time:      " << clean_timer.getTime() << endl;
  cout << "Dedispersion time:       " << dedisp_timer.getTime() << endl;
  cout << "Copy time:               " << copy_timer.getTime() << endl;
  cout << "Baselining time:         " << baseline_timer.getTime() << endl;
  cout << "Normalisation time:      " << normalise_timer.getTime() << endl;
  cout << "Filtering time:          " << filter_timer.getTime() << endl;
  cout << "Find giants time:        " << giants_timer.getTime() << endl;
  cout << "Process candidates time: " << candidates_timer.getTime() << endl;
  cout << "Total time:              " << total_timer.getTime() << endl;
  }

  hd_float time_sum = (memory_timer.getTime() +
                       clean_timer.getTime() +
                       dedisp_timer.getTime() +
                       copy_timer.getTime() +
                       baseline_timer.getTime() +
                       normalise_timer.getTime() +
                       filter_timer.getTime() +
                       giants_timer.getTime() +
                       candidates_timer.getTime());
  hd_float misc_time = total_timer.getTime() - time_sum;
  
  /*
  std::ofstream timing_file("timing.dat", std::ios::app);
  timing_file << total_timer.getTime() << "\t"
              << misc_time << "\t"
              << memory_timer.getTime() << "\t"
              << clean_timer.getTime() << "\t"
              << dedisp_timer.getTime() << "\t"
              << copy_timer.getTime() << "\t"
              << baseline_timer.getTime() << "\t"
              << normalise_timer.getTime() << "\t"
              << filter_timer.getTime() << "\t"
              << giants_timer.getTime() << "\t"
              << candidates_timer.getTime() << endl;
  timing_file.close();
  */
  
#endif // HD_BENCHMARK
  
  if( too_many_giants ) {
    return HD_TOO_MANY_EVENTS;
  }
  else {
    return HD_NO_ERROR;
  }
}

void hd_destroy_pipeline(hd_pipeline pipeline) {
  if( pipeline->params.verbosity >= 2 ) {
    cout << "\tDeleting pipeline object..." << endl;
  }
  
  dedisp_destroy_plan(pipeline->dedispersion_plan);
  
  // Note: This assumes memory owned by pipeline cleans itself up
  if( pipeline ) {
    delete pipeline;
  }
}
