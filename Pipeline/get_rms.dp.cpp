/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/get_rms.h"
#include "hd/median_filter.h"
#include "hd/utils/wrappers.dp.hpp"
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/lambda.hpp>

template <typename T>
struct absolute_val {
    inline auto operator()() const {
        using boost::compute::lambda::_1;
        using boost::compute::lambda::abs;
        return abs(_1);
    }
};

class GetRMSPlan_impl {
    device_vector_wrapper<hd_float> buf1;
    device_vector_wrapper<hd_float> buf2;

public:
	hd_float exec(boost::compute::buffer_iterator<hd_float> d_data, hd_size count) {
        boost::compute::buffer_iterator<hd_float> d_data_begin(d_data);

        // This algorithm works by taking the absolute values of the data
		//   and then repeatedly scrunching them using median-of-5 in order
		//   to approximate the median absolute deviation. The RMS is then
		//   just 1.4862 times this.
		
		buf1.resize(count);
		buf2.resize(count/5);
        boost::compute::buffer_iterator<hd_float> buf1_ptr = buf1.begin();
        boost::compute::buffer_iterator<hd_float> buf2_ptr = buf2.begin();

        boost::compute::transform(d_data_begin, d_data_begin + count, buf1.begin(),
                               absolute_val<hd_float>()());

        for( hd_size size=count; size>1; size/=5 ) {
			median_scrunch5(buf1_ptr, size, buf2_ptr);
			std::swap(buf1_ptr, buf2_ptr);
		}
		// Note: Result is now at buf1_ptr
        boost::compute::buffer_iterator<hd_float> buf1_begin(buf1_ptr);
        hd_float med_abs_dev = buf1_begin[0];
		hd_float rms = med_abs_dev * 1.4862;
		
		return rms;
	}
};

// Public interface (wrapper for implementation)
GetRMSPlan::GetRMSPlan()
	: m_impl(new GetRMSPlan_impl) {}
hd_float GetRMSPlan::exec(boost::compute::buffer_iterator<hd_float> d_data, hd_size count) {
	return m_impl->exec(d_data, count);
}

// Convenience functions for one-off calls
hd_float get_rms(boost::compute::buffer_iterator<hd_float> d_data, hd_size count) {
	return GetRMSPlan().exec(d_data, count);
}
hd_error normalise(boost::compute::buffer_iterator<hd_float> d_data, hd_size count)
{
        boost::compute::buffer_iterator<hd_float> d_data_begin(d_data);
        boost::compute::buffer_iterator<hd_float> d_data_end(d_data + count);

        hd_float rms = get_rms(d_data, count);
        boost::compute::transform(d_data_begin, d_data_end,
                           boost::compute::make_constant_iterator(hd_float(1.0) / rms),
                           d_data_begin,
                           boost::compute::multiplies<hd_float>());

        return HD_NO_ERROR;
}
