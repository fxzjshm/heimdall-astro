/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <vector>
#include <memory>
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
#include "hd/utils.dp.hpp"

#include <dedisp.h>

#include <boost/compute/algorithm.hpp>
#include <boost/compute/lambda.hpp>

#define HD_BENCHMARK

#ifdef HD_BENCHMARK
  void start_timer(Stopwatch& timer) { timer.start(); }
  void stop_timer(Stopwatch &timer) {
   boost::compute::system::default_queue().finish();
   timer.stop();
  }
#else
  void start_timer(Stopwatch& timer) { }
  void stop_timer(Stopwatch& timer) { }
#endif // HD_BENCHMARK

#include <utility>
#include <cmath>

// for host-vector and device_vector
template<typename T> using host_vector = std::vector<T>;
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
  device_vector<hd_float> d_time_series;
  device_vector<hd_float> d_filtered_series;
};

hd_error allocate_gpu(const hd_params params) {
  // TODO: This is just a simple proc-->GPU heuristic to get us started
  int gpu_count;
  std::vector<boost::compute::device> devices = boost::compute::system::devices();
  gpu_count = devices.size();
  //int proc_idx;
  //MPI_Comm comm = pl->communicator;
  //MPI_Comm_rank(comm, &proc_idx);
  int proc_idx = params.beam;
  int gpu_idx = params.gpu_id;
  
  if (gpu_idx >= gpu_count) {
    return HD_INVALID_DEVICE_INDEX;
  }
  std::string device_name = devices[gpu_idx].get_info<CL_DEVICE_NAME>();
  setenv("BOOST_COMPUTE_DEFAULT_DEVICE", device_name.c_str(), /* __replace = */ true);
  setenv("BOOST_COMPUTE_DEFAULT_ENFORCE", "1", /* __replace = */ true);
  assert(boost::compute::system::default_device().get_info<CL_DEVICE_NAME>() == device_name);
  if( params.verbosity >= 1 ) {
    cout << "using device " << device_name << endl;
  }
  dedisp_set_device(gpu_idx);
  
  if( params.verbosity >= 1 ) {
    cout << "Process " << proc_idx << " using GPU " << gpu_idx << endl;
  }

  cerr << "yield_cpu is not supported yet in this implemention" << endl;
  /*
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
  */
  
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
  // In OpenCL we should set GPU before creating device memory, otherwise the runtime often crash
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
  pl->d_time_series.resize(series_stride);
  pl->d_filtered_series.resize(series_stride, 0);
  
  stop_timer(memory_timer);
  
  RemoveBaselinePlan          baseline_remover;
  GetRMSPlan                  rms_getter;
  MatchedFilterPlan<hd_float> matched_filter_plan;
  GiantFinder                 giant_finder;

  device_vector_wrapper<hd_float> d_giant_peaks;
  device_vector_wrapper<hd_size> d_giant_inds;
  device_vector_wrapper<hd_size> d_giant_begins;
  device_vector_wrapper<hd_size> d_giant_ends;
  device_vector_wrapper<hd_size> d_giant_filter_inds;
  device_vector_wrapper<hd_size> d_giant_dm_inds;
  device_vector_wrapper<hd_size> d_giant_members;

  typedef boost::compute::buffer_iterator<hd_float> dev_float_ptr;
  typedef boost::compute::buffer_iterator<hd_size> dev_size_ptr;

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
  
  // For each DM
  for( hd_size dm_idx=0; dm_idx<dm_count; ++dm_idx ) {
    hd_size  cur_dm_scrunch = scrunch_factors[dm_idx];
    hd_size  cur_nsamps  = nsamps_computed / cur_dm_scrunch;
    hd_float cur_dt      = pl->params.dt * cur_dm_scrunch;
    
    // Bail if the candidate rate is too high
    if( too_many_giants ) {
      break;
    }
    
    if( pl->params.verbosity >= 4 ) {
      cout << "dm_idx     = " << dm_idx << endl;
      cout << "scrunch    = " << scrunch_factors[dm_idx] << endl;
      cout << "cur_nsamps = " << cur_nsamps << endl;
      cout << "dt0        = " << pl->params.dt << endl;
      cout << "cur_dt     = " << cur_dt << endl;
        
      cout << "\tBaselining and normalising each beam..." << endl;
    }

    boost::compute::buffer_iterator<hd_float> time_series = pl->d_time_series.begin();

    // Copy the time series to the device and convert to floats
    hd_size offset = dm_idx * series_stride * pl->params.dm_nbits/8;
    start_timer(copy_timer);
    switch( pl->params.dm_nbits ) {
    case 8:
        boost::compute::copy((unsigned char *)&pl->h_dm_series[offset],
                             (unsigned char *)&pl->h_dm_series[offset] + cur_nsamps,
                             pl->d_time_series.begin());
        break;
    case 16:
        boost::compute::copy((unsigned short *)&pl->h_dm_series[offset],
                             (unsigned short *)&pl->h_dm_series[offset] + cur_nsamps,
                             pl->d_time_series.begin());
        break;
    case 32:
        // Note: 32-bit implies float, not unsigned int
        boost::compute::copy((float *)&pl->h_dm_series[offset],
                             (float *)&pl->h_dm_series[offset] + cur_nsamps,
                             pl->d_time_series.begin());
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
    boost::compute::transform(
        pl->d_time_series.begin(), pl->d_time_series.end(),
        boost::compute::make_constant_iterator(hd_float(1.0) / rms),
        pl->d_time_series.begin(),
        boost::compute::multiplies<hd_float>());
    boost::compute::system::default_queue().finish();
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

/* DPCT_ORIG     hd_float* filtered_series =
 * thrust::raw_pointer_cast(&pl->d_filtered_series[0]);*/
    boost::compute::buffer_iterator<hd_float> filtered_series = pl->d_filtered_series.begin();

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
/* DPCT_ORIG       hd_size rel_min_tscrunch_width =
   std::max(pl->params.min_tscrunch_width / cur_dm_scrunch, hd_size(1));*/
      hd_size rel_min_tscrunch_width =
          std::max(pl->params.min_tscrunch_width / cur_dm_scrunch, hd_size(1));
/* DPCT_ORIG       hd_size rel_tscrunch_width = std::max(2 * rel_filter_width
                                            / rel_min_tscrunch_width,
                                            hd_size(1));*/
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
        boost::compute::transform(
            boost::compute::buffer_iterator<hd_float>(filtered_series),
            boost::compute::buffer_iterator<hd_float>(filtered_series) + cur_nsamps_filtered,
            boost::compute::make_constant_iterator(hd_float(1.0) / rms),
            boost::compute::buffer_iterator<hd_float>(filtered_series),
            boost::compute::multiplies<hd_float>());
      }
      else
      {
        // rescale the filtered time series (RMS ~ sqrt(time))
        boost::compute::constant_iterator<hd_float>
          norm_val_iter(1.0 / sqrt((hd_float)rel_filter_width));
        boost::compute::transform(
            boost::compute::buffer_iterator<hd_float>(filtered_series),
            boost::compute::buffer_iterator<hd_float>(filtered_series) + cur_nsamps_filtered,
            norm_val_iter,
            boost::compute::buffer_iterator<hd_float>(filtered_series),
            boost::compute::multiplies<hd_float>());
      }
      boost::compute::system::default_queue().finish();

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
      
      hd_size rel_cur_filtered_offset = (cur_filtered_offset /
                                         rel_tscrunch_width);
      using boost::compute::lambda::_1;
      boost::compute::transform(
          d_giant_inds.begin() + prev_giant_count, d_giant_inds.end(),
          d_giant_inds.begin() + prev_giant_count,
          /*first_idx +*/ (_1 + rel_cur_filtered_offset) * cur_scrunch);
      boost::compute::transform(
          d_giant_begins.begin() + prev_giant_count, d_giant_begins.end(),
          d_giant_begins.begin() + prev_giant_count,
          /*first_idx +*/ (_1 + rel_cur_filtered_offset) * cur_scrunch);
      boost::compute::transform(
          d_giant_ends.begin() + prev_giant_count, d_giant_ends.end(),
          d_giant_ends.begin() + prev_giant_count,
          /*first_idx +*/ (_1 + rel_cur_filtered_offset) * cur_scrunch);
      boost::compute::system::default_queue().finish();

      d_giant_filter_inds.resize(d_giant_peaks.size(), filter_idx);
      d_giant_dm_inds.resize(d_giant_peaks.size(), dm_idx);
      // Note: This could be used to track total member samples if desired
      d_giant_members.resize(d_giant_peaks.size(), 1);
      
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
  } // End of DM loop

  hd_size giant_count = d_giant_peaks.size();
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
    boost::compute::buffer_iterator<hd_size> d_giant_labels_ptr = d_giant_labels.begin();

    RawCandidatesOnDevice d_giants;
    d_giants.peaks = d_giant_peaks.begin();
    d_giants.inds = d_giant_inds.begin();
    d_giants.begins = d_giant_begins.begin();
    d_giants.ends = d_giant_ends.begin();
    d_giants.filter_inds = d_giant_filter_inds.begin();
    d_giants.dm_inds = d_giant_dm_inds.begin();
    d_giants.members = d_giant_members.begin();

    hd_size filter_count = get_filter_index(pl->params.boxcar_max) + 1;

    if( pl->params.verbosity >= 2 ) {
      cout << "Grouping coincident candidates..." << endl;
    }

    // ConstRawCandidates * const_d_giants = (ConstRawCandidates *) &d_giants;
  
    hd_size label_count;
    error = label_candidate_clusters(giant_count,
                                     d_giants, // *const_d_giants,
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

    RawCandidatesOnDevice d_groups;
    d_groups.peaks = d_group_peaks.begin();
    d_groups.inds = d_group_inds.begin();
    d_groups.begins = d_group_begins.begin();
    d_groups.ends = d_group_ends.begin();
    d_groups.filter_inds = d_group_filter_inds.begin();
    d_groups.dm_inds = d_group_dm_inds.begin();
    d_groups.members = d_group_members.begin();

    merge_candidates(giant_count,
                     d_giant_labels_ptr,
                     d_giants, // *const_d_giants,
                     d_groups);
  
    // Look up the actual DM of each group
    device_vector_wrapper<hd_float> d_dm_list(dm_list, dm_list + dm_count);
    boost::compute::gather(
        d_group_dm_inds.begin(), d_group_dm_inds.end(),
        d_dm_list.begin(), d_group_dms.begin());
    boost::compute::system::default_queue().finish();

    // Device to host transfer of candidates
    // h_group_peaks = d_group_peaks;
    device_to_host_copy(d_group_peaks, h_group_peaks);
    // h_group_inds = d_group_inds;
    device_to_host_copy(d_group_inds, h_group_inds);
    // h_group_begins = d_group_begins;
    device_to_host_copy(d_group_begins, h_group_begins);
    // h_group_ends = d_group_ends;
    device_to_host_copy(d_group_ends, h_group_ends);
    // h_group_filter_inds = d_group_filter_inds;
    device_to_host_copy(d_group_filter_inds, h_group_filter_inds);
    // h_group_dm_inds = d_group_dm_inds;
    device_to_host_copy(d_group_dm_inds, h_group_dm_inds);
    // h_group_members = d_group_members;
    device_to_host_copy(d_group_members, h_group_members);
    // h_group_dms = d_group_dms;
    device_to_host_copy(d_group_dms, h_group_dms);
    //h_group_flags = d_group_flags;
    //device_to_host_copy(d_group_flags, h_group_flags);
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
