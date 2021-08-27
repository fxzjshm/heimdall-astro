#pragma once

#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/allocator/buffer_allocator.hpp>
#include <boost/compute/container/vector.hpp>

template <class T, class Alloc = boost::compute::buffer_allocator<T>>
class device_vector_wrapper : public boost::compute::vector<T, Alloc> {
public:
    using boost::compute::vector<T, Alloc>::vector;
    typedef Alloc allocator_type;
    typedef typename allocator_type::size_type size_type;

    template <typename OtherAllocator>
    boost::compute::vector<T, Alloc> &operator=(const std::vector<T, OtherAllocator> &v) {
        return boost::compute::vector<T, Alloc>::operator=(v);
    }

    void resize(size_type new_size, const T &x = T()) {
        size_type old_size = boost::compute::vector<T, Alloc>::size();
        boost::compute::vector<T, Alloc>::resize(new_size);
        if(old_size < new_size){
            boost::compute::iota(boost::compute::vector<T, Alloc>::begin() + old_size, boost::compute::vector<T, Alloc>::end(), x);
        }
    }
};


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
