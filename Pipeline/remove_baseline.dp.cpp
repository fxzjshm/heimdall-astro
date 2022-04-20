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

#include "hd/remove_baseline.h"
#include "hd/median_filter.h"
//#include "hd/write_time_series.h"

#include "hd/utils.hpp"
#include <cmath>

class RemoveBaselinePlan_impl {
        device_vector_wrapper<hd_float> buf1;
        device_vector_wrapper<hd_float> buf2;
        device_vector_wrapper<hd_float> baseline;

public:
	hd_error exec(hd_float* d_data, hd_size count,
	              hd_size smooth_radius) {
        heimdall::util::device_pointer<hd_float> d_data_begin(d_data);

        // This algorithm works by scrunching the data down to a time resolution
		//   representative of the desired smoothing length and then stretching
		//   it back out again. The scrunching is done using the median-of-5
		//   to ensure robustness against outliers (e.g., strong RFI spikes).
	
		// Note: This parameter allows tuning to match the smoothing length
		//         of the original iterative-clipping algorithm.
		hd_float oversample = 2;
		// Find the desired time resolution
		hd_size  sample_count =
			(hd_size)(oversample * hd_float(count)/(2*smooth_radius) + 0.5);
		if( sample_count == 0 ) {
			// Too few samples, no need to baseline
			return HD_NO_ERROR;
		}
	
		// As we will use median-of-5, round to sample_count times a power of five
		hd_size nscrunches  = (hd_size)(log(count/sample_count)/log(5.));
        hd_size count_round = std::pow<double>(5., nscrunches) * sample_count;

        buf1.resize(count_round);
		buf2.resize(count_round/5);
        hd_float *buf1_ptr = heimdall::util::get_raw_pointer(&buf1[0]);
        hd_float *buf2_ptr = heimdall::util::get_raw_pointer(&buf2[0]);

        // First we re-sample to the rounded size
		linear_stretch(d_data, count, buf1_ptr, count_round);
	
		// Then we median scrunch until we reach the sample size
		for( hd_size size=count_round; size>sample_count; size/=5 ) {
			median_scrunch5(buf1_ptr, size, buf2_ptr);
			std::swap(buf1_ptr, buf2_ptr);
		}
		// Note: Output is now at buf1_ptr
        heimdall::util::device_pointer<hd_float> buf1_begin(buf1_ptr);
        heimdall::util::device_pointer<hd_float> buf2_begin(buf2_ptr);

        // Then we need to extrapolate the ends
		linear_stretch(buf1_ptr, sample_count, buf2_ptr+1, sample_count*2);
		buf2_begin[0]                = 2*buf2_begin[1] - buf2_begin[2];
		buf2_begin[sample_count*2+1] = (2*buf2_begin[sample_count*2] -
		                                buf2_begin[sample_count*2-1]);
	
		baseline.resize(count);
        hd_float *baseline_ptr = heimdall::util::get_raw_pointer(&baseline[0]);

        // And finally we stretch back to the original length
		linear_stretch(buf2_ptr, sample_count*2+2, baseline_ptr, count);
	
		// TESTING
		//write_device_time_series(d_data, count, 1.f, "pre_baseline.tim");
		//write_device_time_series(baseline_ptr, count, 1.f, "thebaseline.tim");
	
		// Now we just subtract it off
        sycl::impl::transform(
            execution_policy,
            d_data_begin, d_data_begin + count, baseline.begin(),
            d_data_begin,
            std::minus<hd_float>());

        //write_device_time_series(d_data, count, 1.f, "post_baseline.tim");
	
		return HD_NO_ERROR;
	}
};

// Public interface (wrapper for implementation)
RemoveBaselinePlan::RemoveBaselinePlan()
	: m_impl(new RemoveBaselinePlan_impl) {}
hd_error RemoveBaselinePlan::exec(hd_float* d_data, hd_size count,
                                  hd_size smooth_radius) {
	return m_impl->exec(d_data, count, smooth_radius);
}
