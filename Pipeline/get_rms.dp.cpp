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
#include "hd/get_rms.h"
#include "hd/median_filter.h"

/* DPCT_ORIG #include <thrust/device_vector.h>*/
#include <dpct/dpl_utils.hpp>
/* DPCT_ORIG #include <thrust/transform.h>*/

/* DPCT_ORIG #include <thrust/iterator/constant_iterator.h>*/

template <typename T>
/* DPCT_ORIG struct absolute_val : public thrust::unary_function<T,T> {*/
/*
DPCT1044:12: thrust::unary_function was removed because std::unary_function has
been deprecated in C++11. You may need to remove references to typedefs from
thrust::unary_function in the class definition.
*/
struct absolute_val {
/* DPCT_ORIG 	inline __host__ __device__*/
        inline T operator()(T x) const { return abs(x); }
};

class GetRMSPlan_impl {
/* DPCT_ORIG 	thrust::device_vector<hd_float> buf1;*/
        dpct::device_vector<hd_float> buf1;
/* DPCT_ORIG 	thrust::device_vector<hd_float> buf2;*/
        dpct::device_vector<hd_float> buf2;

public:
	hd_float exec(hd_float* d_data, hd_size count) {
/* DPCT_ORIG 		thrust::device_ptr<hd_float> d_data_begin(d_data);*/
                dpct::device_pointer<hd_float> d_data_begin(d_data);

                // This algorithm works by taking the absolute values of the data
		//   and then repeatedly scrunching them using median-of-5 in order
		//   to approximate the median absolute deviation. The RMS is then
		//   just 1.4862 times this.
		
		buf1.resize(count);
		buf2.resize(count/5);
/* DPCT_ORIG 		hd_float* buf1_ptr =
 * thrust::raw_pointer_cast(&buf1[0]);*/
                hd_float *buf1_ptr = dpct::get_raw_pointer(&buf1[0]);
/* DPCT_ORIG 		hd_float* buf2_ptr =
 * thrust::raw_pointer_cast(&buf2[0]);*/
                hd_float *buf2_ptr = dpct::get_raw_pointer(&buf2[0]);

/* DPCT_ORIG 		thrust::transform(d_data_begin, d_data_begin+count,*/
                std::transform(oneapi::dpl::execution::make_device_policy(
                                   dpct::get_default_queue()),
                               d_data_begin, d_data_begin + count, buf1.begin(),
                               absolute_val<hd_float>());

                for( hd_size size=count; size>1; size/=5 ) {
			median_scrunch5(buf1_ptr, size, buf2_ptr);
			std::swap(buf1_ptr, buf2_ptr);
		}
		// Note: Result is now at buf1_ptr
/* DPCT_ORIG 		thrust::device_ptr<hd_float> buf1_begin(buf1_ptr);*/
                dpct::device_pointer<hd_float> buf1_begin(buf1_ptr);
                hd_float med_abs_dev = buf1_begin[0];
		hd_float rms = med_abs_dev * 1.4862;
		
		return rms;
	}
};

// Public interface (wrapper for implementation)
GetRMSPlan::GetRMSPlan()
	: m_impl(new GetRMSPlan_impl) {}
hd_float GetRMSPlan::exec(hd_float* d_data, hd_size count) {
	return m_impl->exec(d_data, count);
}

// Convenience functions for one-off calls
hd_float get_rms(hd_float* d_data, hd_size count) {
	return GetRMSPlan().exec(d_data, count);
}
hd_error normalise(hd_float* d_data, hd_size count)
{
/* DPCT_ORIG 	thrust::device_ptr<hd_float> d_data_begin(d_data);*/
        dpct::device_pointer<hd_float> d_data_begin(d_data);
/* DPCT_ORIG 	thrust::device_ptr<hd_float> d_data_end(d_data + count);*/
        dpct::device_pointer<hd_float> d_data_end(d_data + count);

        hd_float rms = get_rms(d_data, count);
/* DPCT_ORIG 	thrust::transform(d_data_begin, d_data_end,*/
        std::transform(oneapi::dpl::execution::make_device_policy(
                           dpct::get_default_queue()),
                       d_data_begin, d_data_end,
                       /* DPCT_ORIG
                          thrust::make_constant_iterator(hd_float(1.0)/rms),*/
                       dpct::make_constant_iterator(hd_float(1.0) / rms),
                       d_data_begin,
                       /* DPCT_ORIG thrust::multiplies<hd_float>());*/
                       std::multiplies<hd_float>());

        return HD_NO_ERROR;
}
