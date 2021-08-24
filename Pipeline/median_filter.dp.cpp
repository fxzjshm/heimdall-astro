/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "hd/median_filter.h"

#include <boost/compute.hpp>
#include "hd/utils.dp.hpp"

/*
  Note: The implementations of median3-5 here can be derived from
          'sorting networks'.
 */
DEFINE_BOTH_SIDE(median3,
inline float median3(float a, float b, float c) {
	return a < b ? b < c ? b
	                     : a < c ? c : a
	             : a < c ? a
	                     : b < c ? c : b;
}
)

DEFINE_BOTH_SIDE(median4,
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
)

DEFINE_BOTH_SIDE(median5,
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
)

const std::string common_source = type_define_source();

using boost::compute::buffer_iterator;

auto median_filter3_kernel (const buffer_iterator<hd_float> in,
	                        unsigned int count) {
    BOOST_COMPUTE_CLOSURE(hd_float, median_filter3_kernel_closure, (unsigned int i), (in, count), {
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
	});
    return function_with_external_function(median_filter3_kernel_closure, median3_function);
};

auto median_filter5_kernel (const buffer_iterator<hd_float> in,
	                        unsigned int count) {
    BOOST_COMPUTE_CLOSURE(hd_float, median_filter5_kernel_closure, (unsigned int i), (in, count), {
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
    });
    return function_with_external_function(median_filter5_kernel_closure, {median3_function, median4_function, median5_function});
};

auto median_scrunch3_kernel(const buffer_iterator<hd_float> in) {
    BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(hd_float, median_scrunch3_kernel_closure, (unsigned int i), (in), "{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
        hd_float a = in[3*i+0];
		hd_float b = in[3*i+1];
		hd_float c = in[3*i+2];
		return median3(a, b, c);
    ) + "}");
    return function_with_external_function(median_scrunch3_kernel_closure, median3_function);
};

auto median_scrunch5_kernel(const buffer_iterator<hd_float> in) {
    BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(hd_float, median_scrunch5_kernel_closure, (unsigned int i), (in), "{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
        hd_float a = in[5*i+0];
		hd_float b = in[5*i+1];
		hd_float c = in[5*i+2];
		hd_float d = in[5*i+3];
		hd_float e = in[5*i+4];
		return median5(a, b, c, d, e);
	) + "}");
    return function_with_external_function(median_scrunch5_kernel_closure, {median3_function, median4_function, median5_function});
};

auto median_scrunch3_array_kernel(const buffer_iterator<hd_float> in, hd_size size) {
    BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(hd_float, median_scrunch3_array_kernel_closure, (unsigned int i), (in, size), "{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
        hd_size array = i / size;
		hd_size j     = i % size;
		
		hd_float a = in[(3*array+0)*size + j];
		hd_float b = in[(3*array+1)*size + j];
		hd_float c = in[(3*array+2)*size + j];
		return median3(a, b, c);
	) + "}");
    return function_with_external_function(median_scrunch3_array_kernel_closure, median3_function);
};

auto median_scrunch5_array_kernel(const buffer_iterator<hd_float> in, hd_size size) {
    BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(hd_float, median_scrunch5_array_kernel_closure, (unsigned int i), (in, size), "{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
        hd_size array = i / size;
		hd_size j     = i % size;
		
		hd_float a = in[(5*array+0)*size + j];
		hd_float b = in[(5*array+1)*size + j];
		hd_float c = in[(5*array+2)*size + j];
		hd_float d = in[(5*array+3)*size + j];
		hd_float e = in[(5*array+4)*size + j];
		return median5(a, b, c, d, e);
	) + "}");
    return function_with_external_function(median_scrunch5_array_kernel_closure, median5_function);
};

hd_error median_filter3(const buffer_iterator<hd_float> d_in,
                        hd_size                         count,
                        buffer_iterator<hd_float>       d_out)
{

    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);
    using boost::compute::make_counting_iterator;

    boost::compute::transform(
        boost::compute::make_counting_iterator<unsigned int>(0),
        boost::compute::make_counting_iterator<unsigned int>(count),
        d_out_begin, median_filter3_kernel(d_in, count));
    return HD_NO_ERROR;
}

hd_error median_filter5(const buffer_iterator<hd_float> d_in,
                        hd_size count,
                        buffer_iterator<hd_float> d_out) {
    /* DPCT_ORIG 	thrust::device_ptr<hd_float> d_out_begin(d_out);*/
    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);
    using boost::compute::make_counting_iterator;
    boost::compute::transform(
        boost::compute::make_counting_iterator<unsigned int>(0),
        boost::compute::make_counting_iterator<unsigned int>(count),
        d_out_begin, median_filter5_kernel(d_in, count));
    return HD_NO_ERROR;
}

hd_error median_scrunch3(const buffer_iterator<hd_float> d_in,
                         hd_size count,
                         buffer_iterator<hd_float> d_out) {
    /* DPCT_ORIG 	thrust::device_ptr<const hd_float> d_in_begin(d_in);*/
    boost::compute::buffer_iterator<hd_float> d_in_begin(d_in);
    /* DPCT_ORIG 	thrust::device_ptr<hd_float>       d_out_begin(d_out);*/
    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);
    if (count == 1) {
        *d_out_begin = d_in_begin[0];
    } else if (count == 2) {
        *d_out_begin = 0.5f * (d_in_begin[0] + d_in_begin[1]);
    } else {
        // Note: Truncating here is necessary
        hd_size out_count = count / 3;
        using boost::compute::make_counting_iterator;
        boost::compute::transform(
            boost::compute::make_counting_iterator<unsigned int>(0),
            boost::compute::make_counting_iterator<unsigned int>(out_count),
            d_out_begin, median_scrunch3_kernel(d_in));
    }
    return HD_NO_ERROR;
}

hd_error median_scrunch5(buffer_iterator<hd_float> d_in,
                         hd_size count,
                         buffer_iterator<hd_float> d_out) {
    boost::compute::buffer_iterator<hd_float> d_in_begin(d_in);
    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);

    if (count == 1) {
        *d_out_begin = d_in_begin[0];
    } else if (count == 2) {
        *d_out_begin = 0.5f * (d_in_begin[0] + d_in_begin[1]);
    } else if (count == 3) {
        *d_out_begin = median3(d_in_begin[0],
                               d_in_begin[1],
                               d_in_begin[2]);
    } else if (count == 4) {
        *d_out_begin = median4(d_in_begin[0],
                               d_in_begin[1],
                               d_in_begin[2],
                               d_in_begin[3]);
    } else {
        // Note: Truncating here is necessary
        hd_size out_count = count / 5;
        using boost::compute::make_counting_iterator;
        boost::compute::transform(

            boost::compute::make_counting_iterator<unsigned int>(0),
            boost::compute::make_counting_iterator<unsigned int>(out_count),
            d_out_begin, median_scrunch5_kernel(d_in));
    }
    return HD_NO_ERROR;
}

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 3
hd_error median_scrunch3_array(const buffer_iterator<hd_float> d_in,
                               hd_size array_size,
                               hd_size count,
                               buffer_iterator<hd_float> d_out) {
    /* DPCT_ORIG 	thrust::device_ptr<hd_float> d_out_begin(d_out);*/
    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);
    // Note: Truncating here is necessary
    hd_size out_count = count / 3;
    hd_size total = array_size * out_count;
    using boost::compute::make_counting_iterator;
    boost::compute::transform(
        boost::compute::make_counting_iterator<unsigned int>(0),
        boost::compute::make_counting_iterator<unsigned int>(total),
        d_out_begin,
        median_scrunch3_array_kernel(d_in, array_size));
    return HD_NO_ERROR;
}

// Median-scrunches the corresponding elements from a collection of arrays
// Note: This cannot (currently) handle count not being a multiple of 5
hd_error median_scrunch5_array(const buffer_iterator<hd_float> d_in,
                               hd_size array_size,
                               hd_size count,
                               buffer_iterator<hd_float> d_out) {
    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);
    // Note: Truncating here is necessary
    hd_size out_count = count / 5;
    hd_size total = array_size * out_count;
    using boost::compute::make_counting_iterator;
    boost::compute::transform(
        boost::compute::make_counting_iterator<unsigned int>(0),
        boost::compute::make_counting_iterator<unsigned int>(total),
        d_out_begin,
        median_scrunch5_array_kernel(d_in, array_size));
    return HD_NO_ERROR;
}

template <typename T>
inline auto mean2_functor() {
    std::string name = std::string("mean2_functor_") + boost::compute::type_name<T>();
    std::string source = std::string(R"CLC(
        T func_name (T a, T b) {
            return (T)0.5 * (a + b);
        }
    )CLC");
    boost::replace_all(source, "T", boost::compute::type_name<T>());
    boost::replace_all(source, "func_name", name);
    return boost::compute::make_function_from_source<T (T, T)>(name, source);
};

auto mean_scrunch2_array_kernel (const buffer_iterator<hd_float> in, hd_size size) {
    BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(hd_float, mean_scrunch2_array_kernel_closure, (unsigned int i), (in, size), "{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
        hd_size array = i / size;
		hd_size j     = i % size;

		hd_float a = in[(2*array+0)*size + j];
		hd_float b = in[(2*array+1)*size + j];
		return (hd_float)0.5 * (a+b);
    ) + "}");
    return mean_scrunch2_array_kernel_closure;
};

// Note: This can operate 'in-place'
hd_error mean_filter2(const buffer_iterator<hd_float> d_in,
                      hd_size count,
                      buffer_iterator<hd_float> d_out) {
    boost::compute::buffer_iterator<hd_float> d_in_begin(d_in);
    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);
    boost::compute::adjacent_difference(
        d_in_begin, d_in_begin + count, d_out_begin,
        mean2_functor<hd_float>());
    return HD_NO_ERROR;
}

hd_error mean_scrunch2_array(const buffer_iterator<hd_float> d_in,
                             hd_size         array_size,
                             hd_size         count,
                             buffer_iterator<hd_float>       d_out)
{
    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);
    // Note: Truncating here is necessary
    hd_size out_count = count / 2;
    hd_size total = array_size * out_count;
    using boost::compute::make_counting_iterator;
    boost::compute::transform(
        boost::compute::make_counting_iterator<unsigned int>(0),
        boost::compute::make_counting_iterator<unsigned int>(total),
        d_out_begin,
        mean_scrunch2_array_kernel(d_in, array_size));
    return HD_NO_ERROR;
}

// suggested by Ewan Barr (2016 email)
auto linear_stretch_functor2(const buffer_iterator<hd_float> in, unsigned in_size, float step) {
    hd_float correction = ((int)(step/2))/step;

    BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(hd_float, linear_stretch_functor2_closure, (unsigned int out_idx), (in, in_size, step, correction), "{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
        {
                float fidx = ((float)out_idx) / step - correction;
                unsigned idx = (unsigned) fidx;
                if (fidx<0)
                        idx = 0;
                else if (idx + 1 >= in_size)
                        idx = in_size-2;
                return in[idx] + ((in[idx+1] - in[idx]) * (fidx-idx));
        }
    ) + "}");
    return linear_stretch_functor2_closure;
};

auto linear_stretch_functor(const buffer_iterator<hd_float> in,
	                          hd_size in_count, hd_size out_count) {
    hd_float step = (hd_float(in_count-1)/(out_count-1));
    BOOST_COMPUTE_CLOSURE_WITH_SOURCE_STRING(hd_float, linear_stretch_functor_closure, (unsigned int i), (in, step), "{" + common_source + BOOST_COMPUTE_STRINGIZE_SOURCE(
        hd_float     x = i * step;
		unsigned int j = x;
		return in[j] + ((x-j > 1e-5) ? (x-j)*(in[j+1]-in[j]) : 0.f);
    ) + "}");
    return linear_stretch_functor_closure;
};

hd_error linear_stretch(const buffer_iterator<hd_float> d_in,
                        hd_size         in_count,
                        buffer_iterator<hd_float>       d_out,
                        hd_size         out_count)
{
    using boost::compute::make_counting_iterator;
    boost::compute::buffer_iterator<hd_float> d_out_begin(d_out);

    // Ewan found this code to contain a bug, and suggested the latter
    // thrust::transform(make_counting_iterator<unsigned int>(0),
    //                   make_counting_iterator<unsigned int>(out_count),
    //                   d_out_begin,
    //                   linear_stretch_functor(d_in, in_count, out_count));
    boost::compute::transform(

        boost::compute::make_counting_iterator<unsigned int>(0),
        boost::compute::make_counting_iterator<unsigned int>(out_count), d_out_begin,
        linear_stretch_functor2(d_in, in_count,
                                hd_float(out_count - 1) / (in_count - 1)));

    return HD_NO_ERROR;
}
