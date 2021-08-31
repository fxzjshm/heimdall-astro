#pragma once

#include "hd/utils/device_vector_wrapper.dp.hpp"
#include "hd/utils/discard_iterator_wrapper.dp.hpp"

/*
template<class T>
class buffer_iterator_wrapper;
template<class T>
class device_ptr_wrapper;

template<class T>
class device_ptr_wrapper : public boost::compute::detail::device_ptr<T> {
public:
    using boost::compute::detail::device_ptr<T>::device_ptr;

    operator boost::compute::buffer_iterator<T>() {
        return boost::compute::buffer_iterator<T>(boost::compute::detail::device_ptr<T>::get_buffer(),
                                          boost::compute::detail::device_ptr<T>::get_index());
    }
};

template<class T>
class buffer_iterator_wrapper : public boost::compute::buffer_iterator<T> {
public:
    using boost::compute::buffer_iterator<T>::buffer_iterator;
    
    buffer_iterator_wrapper(boost::compute::buffer_iterator<T> iter)
        : boost::compute::buffer_iterator<T>::buffer_iterator(iter) {}
    
    operator boost::compute::detail::device_ptr<T>() {
        return boost::compute::detail::device_ptr<T>(boost::compute::buffer_iterator<T>::get_buffer(),
                                     boost::compute::buffer_iterator<T>::get_index());
    }
};

*/
