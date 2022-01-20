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

#include "hd/median_filter.h"
#include "hd/utils.hpp"

#include <boost/iterator/counting_iterator.hpp>
#include <sycl/algorithm/adjacent_difference.hpp>
#include <sycl/algorithm/transform.hpp>

/*
  Note: The implementations of median3-5 here can be derived from
          'sorting networks'.
 */

inline float median3(float a, float b, float c) {
        return a < b ? b < c ? b
	                      : a < c ? c : a
	             : a < c ? a
	                     : b < c ? c : b;
}

inline float median4(float a, float b, float c, float d) {
        return a < c ? b < d ? a < b ? c < d ? 0.5f*(b+c) : 0.5f*(b+d)
	                             : c < d ? 0.5f*(a+c) : 0.5f*(a+d)
	                     : a < d ? c < b ? 0.5f*(d+c) : 0.5f*(b+d)
	                             : c < b ? 0.5f*(a+c) : 0.5f*(a+b)
	             : b < d ? c < b ? a < d ? 0.5f*(b+a) : 0.5f*(b+d)
	                             : a < d ? 0.5f*(a+c) : 0.5f*(c+d)
	                     : c < d ? a < b ? 0.5f*(d+a) : 0.5f*(b+d)
	                             : a < b ? 0.5f*(a+c) : 0.5f*(c+b);
}

inline float median5(float a, float b, float c, float d, float e) {
    // Note: This wicked code is by 'DRBlaise' and was found here:
	//         http://stackoverflow.com/a/2117018
	return b < a ? d < c ? b < d ? a < e ? a < d ? e < d ? e : d
                                                 : c < a ? c : a
                                         : e < d ? a < d ? a : d
                                                 : c < e ? c : e
                                 : c < e ? b < c ? a < c ? a : c
                                                 : e < b ? e : b
                                         : b < e ? a < e ? a : e
                                                 : c < b ? c : b
                         : b < c ? a < e ? a < c ? e < c ? e : c
                                                 : d < a ? d : a
                                         : e < c ? a < c ? a : c
                                                 : d < e ? d : e
                                 : d < e ? b < d ? a < d ? a : d
                                                 : e < b ? e : b
                                         : b < e ? a < e ? a : e
                                                 : d < b ? d : b
                 : d < c ? a < d ? b < e ? b < d ? e < d ? e : d
                                                 : c < b ? c : b
                                         : e < d ? b < d ? b : d
                                                 : c < e ? c : e
                                 : c < e ? a < c ? b < c ? b : c
                                                 : e < a ? e : a
                                         : a < e ? b < e ? b : e
                                                 : c < a ? c : a
                         : a < c ? b < e ? b < c ? e < c ? e : c
                                                 : d < b ? d : b
                                         : e < c ? b < c ? b : c
                                                 : d < e ? d : e
                                 : d < e ? a < d ? b < d ? b : d
                                                 : e < a ? e : a
                                         : a < e ? b < e ? b : e
                                                 : d < a ? d : a;
}

struct median_filter3_kernel {
    const hd_float* in;
	unsigned int    count;
	median_filter3_kernel(const hd_float* in_,
	                      unsigned int count_)
		: in(in_), count(count_) {}
        inline hd_float operator()(unsigned int i) const {
                // Note: We shrink the window near boundaries
		if( i > 0 && i < count-1 ) {
			return median3(in[i-1], in[i], in[i+1]);
		}
		else if( i == 0 ) {
			return 0.5f*(in[i]+in[i+1]);
		}
		else { //if( i == count-1 ) {
			return 0.5f*(in[i]+in[i-1]);
		}
	}
};

struct median_filter5_kernel {
    const hd_float* in;
	unsigned int    count;
	median_filter5_kernel(const hd_float* in_,
	                      unsigned int count_)
		: in(in_), count(count_) {}
        inline hd_float operator()(unsigned int i) const {
        // Note: We shrink the window near boundaries
		if( i > 1 && i < count-2 ) {
			return median5(in[i-2], in[i-1], in[i], in[i+1], in[i+2]);
		}
		else if( i == 0 ) {
			return median3(in[i], in[i+1], in[i+2]);
		}
		else if( i == 1 ) {
			return median4(in[i-1], in[i], in[i+1], in[i+2]);
		}
		else if( i == count-1 ) {
			return median3(in[i], in[i-1], in[i-2]);
		}
		else { //if ( i == count-2 ) {
			return median4(in[i+1], in[i], in[i-1], in[i-2]);
		}
	}
};

struct median_scrunch3_kernel {
    const hd_float* in;
	median_scrunch3_kernel(const hd_float* in_)
		: in(in_) {}

    inline hd_float operator()(unsigned int i) const {
        hd_float a = in[3*i+0];
		hd_float b = in[3*i+1];
		hd_float c = in[3*i+2];
		return median3(a, b, c);
	}
};

struct median_scrunch5_kernel {
    const hd_float* in;
	median_scrunch5_kernel(const hd_float* in_)
		: in(in_) {}

        inline hd_float operator()(unsigned int i) const {
                hd_float a = in[5*i+0];
		hd_float b = in[5*i+1];
		hd_float c = in[5*i+2];
		hd_float d = in[5*i+3];
		hd_float e = in[5*i+4];
		return median5(a, b, c, d, e);
	}
};

struct median_scrunch3_array_kernel {
    const hd_float* in;
	const hd_size   size;
	median_scrunch3_array_kernel(const hd_float* in_, hd_size size_)
		: in(in_), size(size_) {}

        inline hd_float operator()(unsigned int i) const {
                hd_size array = i / size;
		hd_size j     = i % size;
		
		hd_float a = in[(3*array+0)*size + j];
		hd_float b = in[(3*array+1)*size + j];
		hd_float c = in[(3*array+2)*size + j];
		return median3(a, b, c);
	}
};

struct median_scrunch5_array_kernel {
    const hd_float* in;
	const hd_size   size;
	median_scrunch5_array_kernel(const hd_float* in_, hd_size size_)
		: in(in_), size(size_) {}

        inline hd_float operator()(unsigned int i) const {
                hd_size array = i / size;
		hd_size j     = i % size;
		
		hd_float a = in[(5*array+0)*size + j];
		hd_float b = in[(5*array+1)*size + j];
		hd_float c = in[(5*array+2)*size + j];
		hd_float d = in[(5*array+3)*size + j];
		hd_float e = in[(5*array+4)*size + j];
		return median5(a, b, c, d, e);
	}
};

hd_error median_filter3(const hd_float* d_in,
                        hd_size         count,
                        hd_float*       d_out)
{
    
    dpct::device_pointer<hd_float> d_out_begin(d_out);
    using boost::iterators::make_counting_iterator;

    sycl::impl::transform(execution_policy,
        make_counting_iterator<unsigned int>(0),
        make_counting_iterator<unsigned int>(count),
        d_out_begin, median_filter3_kernel(d_in, count));
    return HD_NO_ERROR;
}

hd_error median_filter5(const hd_float* d_in,
                        hd_size         count,
                        hd_float*       d_out)
{
    dpct::device_pointer<hd_float> d_out_begin(d_out);
    using boost::iterators::make_counting_iterator;
    sycl::impl::transform(execution_policy,
        make_counting_iterator<unsigned int>(0),
        make_counting_iterator<unsigned int>(count),
        d_out_begin, median_filter5_kernel(d_in, count));
    return HD_NO_ERROR;
}

hd_error median_scrunch3(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out)
{
    dpct::device_pointer<const hd_float> d_in_begin(d_in);
    dpct::device_pointer<hd_float> d_out_begin(d_out);
    if( count == 1 ) {
		*d_out_begin = d_in_begin[0];
	}
	else if( count == 2 ) {
		*d_out_begin = 0.5f*(d_in_begin[0] + d_in_begin[1]);
	}
	else {
		// Note: Truncating here is necessary
		hd_size out_count = count / 3;
		using boost::iterators::make_counting_iterator;
        sycl::impl::transform(execution_policy,
            make_counting_iterator<unsigned int>(0),
            make_counting_iterator<unsigned int>(out_count),
            d_out_begin, median_scrunch3_kernel(d_in));
    }
	return HD_NO_ERROR;
}

hd_error median_scrunch5(const hd_float* d_in,
                         hd_size         count,
                         hd_float*       d_out)
{
    dpct::device_pointer<const hd_float> d_in_begin(d_in);
    dpct::device_pointer<hd_float> d_out_begin(d_out);

    if( count == 1 ) {
		*d_out_begin = d_in_begin[0];
	}
	else if( count == 2 ) {
		*d_out_begin = 0.5f*(d_in_begin[0] + d_in_begin[1]);
	}
	else if( count == 3 ) {
		*d_out_begin = median3(d_in_begin[0],
		                       d_in_begin[1],
		                       d_in_begin[2]);
	}
	else if( count == 4 ) {
		*d_out_begin = median4(d_in_begin[0],
		                       d_in_begin[1],
		                       d_in_begin[2],
		                       d_in_begin[3]);
	}
	else {
		// Note: Truncating here is necessary
		hd_size out_count = count / 5;
		using boost::iterators::make_counting_iterator;
        sycl::impl::transform(execution_policy,
            make_counting_iterator<unsigned int>(0),
            make_counting_iterator<unsigned int>(out_count),
            d_out_begin, median_scrunch5_kernel(d_in));
        }
	return HD_NO_ERROR;
}

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 3
hd_error median_scrunch3_array(const hd_float* d_in,
                               hd_size         array_size,
                               hd_size         count,
                               hd_float*       d_out)
{
    dpct::device_pointer<hd_float> d_out_begin(d_out);
    // Note: Truncating here is necessary
	hd_size out_count = count / 3;
	hd_size total     = array_size * out_count;
	using boost::iterators::make_counting_iterator;
    sycl::impl::transform(execution_policy,
        make_counting_iterator<unsigned int>(0),
        make_counting_iterator<unsigned int>(total),
        d_out_begin,
        median_scrunch3_array_kernel(d_in, array_size));
    return HD_NO_ERROR;
}

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 5
hd_error median_scrunch5_array(const hd_float* d_in,
                               hd_size         array_size,
                               hd_size         count,
                               hd_float*       d_out)
{
    dpct::device_pointer<hd_float> d_out_begin(d_out);
    // Note: Truncating here is necessary
	hd_size out_count = count / 5;
	hd_size total     = array_size * out_count;
	using boost::iterators::make_counting_iterator;
    sycl::impl::transform(execution_policy,
        make_counting_iterator<unsigned int>(0),
        make_counting_iterator<unsigned int>(total),
        d_out_begin,
        median_scrunch5_array_kernel(d_in, array_size));
    return HD_NO_ERROR;
}

template <typename T>
struct mean2_functor {
    inline T operator()(T a, T b) const { return (T)0.5 * (a + b); }
};

struct mean_scrunch2_array_kernel {
    const hd_float* in;
	const hd_size   size;
	mean_scrunch2_array_kernel(const hd_float* in_, hd_size size_)
		: in(in_), size(size_) {}
    inline hd_float operator()(unsigned int i) const {
        hd_size array = i / size;
		hd_size j     = i % size;
		
		hd_float a = in[(2*array+0)*size + j];
		hd_float b = in[(2*array+1)*size + j];
		return (hd_float)0.5 * (a+b);
	}
};

// Note: This can operate 'in-place'
hd_error mean_filter2(const hd_float* d_in,
                      hd_size         count,
                      hd_float*       d_out)
{
    dpct::device_pointer<const hd_float> d_in_begin(d_in);
    dpct::device_pointer<hd_float> d_out_begin(d_out);
    sycl::impl::adjacent_difference(execution_policy,
        // d_in_begin, d_in_begin + count, d_out_begin,
        d_in, d_in + count, d_out,
        mean2_functor<hd_float>());
    return HD_NO_ERROR;
}

hd_error mean_scrunch2_array(const hd_float* d_in,
                             hd_size         array_size,
                             hd_size         count,
                             hd_float*       d_out)
{
    dpct::device_pointer<hd_float> d_out_begin(d_out);
    // Note: Truncating here is necessary
	hd_size out_count = count / 2;
	hd_size total     = array_size * out_count;
	using boost::iterators::make_counting_iterator;
    sycl::impl::transform(execution_policy,
        make_counting_iterator<unsigned int>(0),
        make_counting_iterator<unsigned int>(total),
        d_out_begin,
        mean_scrunch2_array_kernel(d_in, array_size));
    return HD_NO_ERROR;
}

// suggested by Ewan Barr (2016 email)
struct linear_stretch_functor2 {
        const hd_float* in;
        unsigned in_size;
        float step;
        float correction;

        linear_stretch_functor2(const hd_float* in_, unsigned in_size, float step)
          : in(in_), in_size(in_size), step(step), correction(((int)(step/2))/step){}

        inline hd_float operator()(unsigned out_idx) const
        {
                float fidx = ((float)out_idx) / step - correction;
                unsigned idx = (unsigned) fidx;
                if (fidx<0)
                        idx = 0;
                else if (idx + 1 >= in_size)
                        idx = in_size-2;
                return in[idx] + ((in[idx+1] - in[idx]) * (fidx-idx));
        }
};

struct linear_stretch_functor {
    const hd_float* in;
	hd_float        step;
	linear_stretch_functor(const hd_float* in_,
	                       hd_size in_count, hd_size out_count)
		: in(in_), step(hd_float(in_count-1)/(out_count-1)) {}
    inline hd_float operator()(unsigned int i) const {
                hd_float     x = i * step;
		unsigned int j = x;
		return in[j] + ((x-j > 1e-5) ? (x-j)*(in[j+1]-in[j]) : 0.f);
	}
};

hd_error linear_stretch(const hd_float* d_in,
                        hd_size         in_count,
                        hd_float*       d_out,
                        hd_size         out_count)
{
	using boost::iterators::make_counting_iterator;
    dpct::device_pointer<hd_float> d_out_begin(d_out);

    // Ewan found this code to contain a bug, and suggested the latter
	// thrust::transform(make_counting_iterator<unsigned int>(0),
	//                   make_counting_iterator<unsigned int>(out_count),
	//                   d_out_begin,
	//                   linear_stretch_functor(d_in, in_count, out_count));
    sycl::impl::transform(execution_policy,
        make_counting_iterator<unsigned int>(0),
        make_counting_iterator<unsigned int>(out_count), d_out_begin,
        linear_stretch_functor2(d_in, in_count,
                                hd_float(out_count - 1) / (in_count - 1)));

    return HD_NO_ERROR;
}
