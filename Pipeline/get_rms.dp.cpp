/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/get_rms.h"
#include "hd/median_filter.h"
//#include "hd/write_time_series.h"
#include "hd/utils/wrappers.dp.hpp"
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/lambda.hpp>

template <typename T, bool is_integral = std::is_integral<T>::value, bool is_floating_point = std::is_floating_point<T>::value>
struct absolute_val;

template <typename T>
struct absolute_val<T, true, false> {
    inline auto operator()() const {
        using boost::compute::lambda::_1;
        using boost::compute::lambda::abs;
        return abs(_1);
    }
};

template <typename T>
struct absolute_val <T, false, true> {
    inline auto operator()() const {
        using boost::compute::lambda::_1;
        using boost::compute::lambda::fabs;
        return fabs(_1);
    }
};

class GetRMSPlan_impl {
    device_vector_wrapper<hd_float> buf1;
    device_vector_wrapper<hd_float> buf2;

public:
	hd_float exec(boost::compute::buffer_iterator<hd_float> d_data, hd_size count, boost::compute::command_queue& queue) {
        boost::compute::buffer_iterator<hd_float> d_data_begin(d_data);

        // This algorithm works by taking the absolute values of the data
		//   and then repeatedly scrunching them using median-of-5 in order
		//   to approximate the median absolute deviation. The RMS is then
		//   just 1.4862 times this.
		
		buf1.resize(count, queue);
		buf2.resize(count/5, queue);
        boost::compute::buffer_iterator<hd_float> buf1_ptr = buf1.begin();
        boost::compute::buffer_iterator<hd_float> buf2_ptr = buf2.begin();

        boost::compute::transform(d_data_begin, d_data_begin + count, buf1.begin(),
                               absolute_val<hd_float>()(), queue);
        queue.finish();

        for( hd_size size=count; size>1; size/=5 ) {
			median_scrunch5(buf1_ptr, size, buf2_ptr, queue);
            // write_device_time_series(buf1_ptr, size, 1.f, "median_scrunch5_size" + std::to_string(size) + "_in.tim");
            // write_device_time_series(buf2_ptr, size/5, 1.f, "median_scrunch5_size" + std::to_string(size) + "_out.tim");
			std::swap(buf1_ptr, buf2_ptr);
		}
		// Note: Result is now at buf1_ptr
        boost::compute::buffer_iterator<hd_float> buf1_begin(buf1_ptr);
        hd_float med_abs_dev = buf1_begin.read(queue);
		hd_float rms = med_abs_dev * 1.4862;
		
		return rms;
	}
};

// Public interface (wrapper for implementation)
GetRMSPlan::GetRMSPlan()
	: m_impl(new GetRMSPlan_impl) {}
hd_float GetRMSPlan::exec(boost::compute::buffer_iterator<hd_float> d_data, hd_size count, boost::compute::command_queue& queue) {
	return m_impl->exec(d_data, count, queue);
}

// Convenience functions for one-off calls
hd_float get_rms(boost::compute::buffer_iterator<hd_float> d_data, hd_size count, boost::compute::command_queue& queue) {
	return GetRMSPlan().exec(d_data, count, queue);
}
hd_error normalise(boost::compute::buffer_iterator<hd_float> d_data, hd_size count, boost::compute::command_queue& queue)
{
        boost::compute::buffer_iterator<hd_float> d_data_begin(d_data);
        boost::compute::buffer_iterator<hd_float> d_data_end(d_data + count);

        hd_float rms = get_rms(d_data, count, queue);
        boost::compute::transform(d_data_begin, d_data_end,
                           boost::compute::make_constant_iterator(argument_wrapper("coeff", hd_float(1.0) / rms)),
                           d_data_begin,
                           boost::compute::multiplies<hd_float>(),
                           queue);
        queue.finish();

        return HD_NO_ERROR;
}
