#pragma once

#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/allocator/buffer_allocator.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/system.hpp>

template <class T, class Alloc = boost::compute::buffer_allocator<T>>
class device_vector_wrapper : public boost::compute::vector<T, Alloc> {
public:
    using boost::compute::vector<T, Alloc>::vector;
    typedef Alloc allocator_type;
    typedef typename allocator_type::size_type size_type;
    using super = boost::compute::vector<T, Alloc>;

    template <typename OtherAllocator>
    boost::compute::vector<T, Alloc> &operator=(const std::vector<T, OtherAllocator> &v) {
        return super::operator=(v);
    }

    inline void resize(size_type size, boost::compute::command_queue& queue = boost::compute::system::default_queue()) {
        super::resize(size, queue);
    }

    void resize(size_type new_size, const T &x, boost::compute::command_queue& queue = boost::compute::system::default_queue()) {
        size_type old_size = super::size();
        super::resize(new_size);
        if(old_size < new_size){
            boost::compute::fill(super::begin() + old_size, super::end(), x);
            queue.finish();
        }
    }
};