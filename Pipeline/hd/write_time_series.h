/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#pragma once

#include <vector>
#include <string>
using std::string;
#include <fstream>
#include <iostream>

#include "hd/utils/buffer_iterator.dp.hpp"
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/system.hpp>

namespace detail {
// TODO: These were copied from header.hpp. Not sure if this is a good idea.
// Write string value
template<class String, class BinaryStream>
void header_write(BinaryStream& stream, const String& str) {
	std::string s = str;
	int len = s.size();
	// TODO: Apply byte swapping for endian-correctness
	stream.write((char*)&len, sizeof(int));
	// TODO: Apply byte swapping for endian-correctness
	stream.write(s.c_str(), len*sizeof(char));
}

// Write integer value
template<class String, class BinaryStream>
void header_write(BinaryStream& stream, String name, int val) {
	header_write(stream, name);
	// TODO: Apply byte swapping for endian-correctness
	stream.write((char*)&val, sizeof(int));
}

// Write floating-point value
template<class String, class BinaryStream>
void header_write(BinaryStream& stream, String name, double val) {
	header_write(stream, name);
	// TODO: Apply byte swapping for endian-correctness
	stream.write((char*)&val, sizeof(double));
}

// Write coordinates
template<class BinaryStream>
void header_write(BinaryStream& stream,
				  double raj, double dej,
				  double az, double za) {
	header_write(stream, "src_raj",  raj);
	header_write(stream, "src_dej",  dej);
	header_write(stream, "az_start", az);
	header_write(stream, "za_start", za);
}

inline
void write_time_series_header(size_t nbits, float dt, std::ofstream& out_file) {
	// Write the required header information
	header_write(out_file, "HEADER_START");
	//header_write(out_file, "telescope_id", header.telescope_id);
	//header_write(out_file, "machine_id", header.machine_id);
	//header_write(out_file,
	//			 header.src_raj, header.src_dej,
	//			 header.az_start, header.za_start);
	header_write(out_file, "data_type", 2);
	//header_write(out_file, "refdm", header.refdm);
	//header_write(out_file, "fch1", header.f0);
	//header_write(out_file, "barycentric", 6);//header.barycentric);
	header_write(out_file, "nchans", 1);//nbands);
	header_write(out_file, "nbits", (int)nbits);
	//header_write(out_file, "tstart", 0.f);//header.tstart);
	header_write(out_file, "tsamp", dt);
	header_write(out_file, "nifs", 1);//header.nifs);
	header_write(out_file, "HEADER_END");
}
} // namespace detail

// Float data type
inline
void write_host_time_series(const float* data,
							size_t       nsamps,
                            float        dt,
							string       filename)
{
    std::cout << "write_host_time_series: " << filename << std::endl;
	// Open the output file and write the data
	std::ofstream file(filename.c_str(), std::ios::binary);
	detail::write_time_series_header(32, dt, file);
	size_t size_bytes = nsamps*sizeof(float);
	file.write((char*)data, size_bytes);
	file.close();

    std::ofstream file2((filename + ".txt").c_str());
    for(size_t i=0;i<nsamps;i++){
        file2<<data[i]<<' ';
    }
    file2<<std::endl;
    file2.close();
}

/*
template<class Iterator>
inline
void write_device_time_series(const Iterator data,
                              size_t      nsamps,
                              float       dt,
                              string      filename,
                              typename std::enable_if<
                                  std::is_same<typename std::iterator_traits<Iterator>::value_type, float>::value
                              >::type* = 0)
{
	std::vector<float> h_data(nsamps);
    // boost::compute::system::default_queue().memcpy(&h_data[0], data, nsamps * sizeof(float)).wait();
    boost::compute::copy(data, data + nsamps, h_data.begin());
	write_host_time_series(&h_data[0], nsamps, dt, filename);
}
*/

// Integer data type
inline
void write_host_time_series(const unsigned int* data,
                            size_t      nsamps,
                            size_t      nbits,
                            float       dt,
                            string      filename)
{
	// Here we convert the data to floats before writing the data
	std::vector<float> float_data(nsamps);
	switch( nbits ) {
	case sizeof(char)*8:
		for( int i=0; i<(int)nsamps; ++i )
			float_data[i] = (float)((unsigned char*)data)[i];
		break;
	case sizeof(short)*8:
		for( int i=0; i<(int)nsamps; ++i )
			float_data[i] = (float)((unsigned short*)data)[i];
		break;
	case sizeof(int)*8:
		for( int i=0; i<(int)nsamps; ++i )
			float_data[i] = (float)((unsigned int*)data)[i];
		break;
	/*
	case sizeof(long long)*8:
		for( int i=0; i<(int)nsamps; ++i )
			float_data[i] = (float)((unsigned long long*)data)[i];
	*/
	default:
		// Unpack to float
		size_t mask = (1 << nbits) - 1;
		size_t spw = sizeof(unsigned int)*8 / nbits; // Samples per word
		for( int i=0; i<(int)nsamps; ++i )
			float_data[i] = (data[i/spw] >> (i % spw * nbits)) & mask;
	}
	write_host_time_series(&float_data[0], nsamps, dt, filename);
}

template<class Iterator>
inline
void write_device_time_series(const Iterator data,
                              size_t      nsamps,
                              size_t      nbits,
                              float       dt,
                              string      filename,
                              typename std::enable_if<
                                  std::is_same<typename std::iterator_traits<Iterator>::value_type, unsigned int>::value
                              >::type* = 0)
{
	size_t nsamps_words = nsamps * nbits/(sizeof(unsigned int)*8);
	std::vector<unsigned int> h_data(nsamps_words);
    boost::compute::copy(data, data + nsamps_words, h_data.begin());
	write_host_time_series(&h_data[0], nsamps, nbits, dt, filename);
}

template<class Iterator>
inline
void write_device_time_series(const Iterator data,
                              size_t      nsamps,
                              float       dt,
                              string      filename)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;
	std::vector<T> h_data(nsamps);
    // boost::compute::system::default_queue().memcpy(&h_data[0], data, nsamps * sizeof(float)).wait();
    boost::compute::copy(data, data + nsamps, h_data.begin());
    if constexpr (std::is_same<T, float>::value) {
        write_host_time_series(&h_data[0], nsamps, dt, filename);
    } else {
        std::vector<float> h_data_float(nsamps);
        for (size_t i = 0; i < nsamps; i++) {
            h_data_float[i] = static_cast<float>(h_data[i]);
        }
	    write_host_time_series(&h_data_float[0], nsamps, dt, filename);
    }
}

template<typename Vector>
inline void write_vector(const Vector& v, std::string filename) {
    std::cout<<"write_vector: "<<filename<<": ";
    typedef typename Vector::value_type T;
    std::vector<T> data;
    data.resize(v.size());
    boost::compute::copy(v.begin(), v.end(), data.begin());
    boost::compute::system::default_queue().finish();
    std::ofstream file2((filename + ".txt").c_str());
    for(size_t i=0;i<data.size();i++){
        file2<<data[i]<<' ';
        if(v.size()<100)std::cout<<data[i]<<' ';
    }
    file2<<std::endl;
    std::cout<<std::endl;
    file2.close();
}