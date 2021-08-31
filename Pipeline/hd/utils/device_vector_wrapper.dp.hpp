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

    inline void resize(size_type size) {
        boost::compute::vector<T, Alloc>::resize(size);
    }

    void resize(size_type new_size, const T &x) {
        size_type old_size = boost::compute::vector<T, Alloc>::size();
        boost::compute::vector<T, Alloc>::resize(new_size);
        if(old_size < new_size){
            boost::compute::iota(boost::compute::vector<T, Alloc>::begin() + old_size, boost::compute::vector<T, Alloc>::end(), x);
        }
    }
};