#include "hd/utils/buffer_iterator.dp.hpp"

namespace boost::compute {

template<typename A, typename B>
inline bool operator==(const boost::compute::buffer_iterator<A> a, const boost::compute::buffer_iterator<B> b) {
    return (a.get_buffer() == b.get_buffer()) && (a.get_index() == b.get_index());
}

}