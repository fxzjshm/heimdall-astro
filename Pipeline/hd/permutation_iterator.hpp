#pragma once

#include <type_traits>
#include <iterator>

namespace heimdall {
namespace util {

/**
 *  @brief  Permutation iterator. Modfied from ostream_iterator and ZipIter
*/
template<class ElementIterator, class IndexIterator>
class permutation_iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = typename std::iterator_traits<ElementIterator>::difference_type;
    using value_type        = typename std::iterator_traits<ElementIterator>::value_type;
    using pointer           = typename std::iterator_traits<ElementIterator>::pointer;
    using reference         = typename std::iterator_traits<ElementIterator>::reference;

    permutation_iterator() = default;
    permutation_iterator(const permutation_iterator &rhs) = default;
    permutation_iterator(permutation_iterator&& rhs) = default;
    permutation_iterator(ElementIterator element_iterator_, IndexIterator index_iterator_)
      : element_iterator(element_iterator_), index_iterator(index_iterator_) {}

    permutation_iterator& operator=(const permutation_iterator& rhs) = default;
    permutation_iterator& operator=(permutation_iterator&& rhs) = default;

    permutation_iterator& operator+=(const difference_type d) {
      index_iterator += d;
      return *this;
    }
    permutation_iterator& operator-=(const difference_type d) { return operator+=(-d); }

    reference operator* () const { return *(element_iterator + *(index_iterator)); }
    //pointer   operator->() const { return (element_iterator + *(index_iterator)).operator->(); }
    reference operator[](difference_type rhs) const {return *(operator+(rhs));}

    permutation_iterator& operator++() { return operator+=( 1); }
    permutation_iterator& operator--() { return operator+=(-1); }
    permutation_iterator operator++(int) {permutation_iterator tmp(*this); operator++(); return tmp;}
    permutation_iterator operator--(int) {permutation_iterator tmp(*this); operator--(); return tmp;}

    difference_type operator-(const permutation_iterator& rhs) const {return index_iterator - rhs.index_iterator;}
    permutation_iterator operator+(const difference_type d) const {permutation_iterator tmp(*this); tmp += d; return tmp;}
    permutation_iterator operator-(const difference_type d) const {permutation_iterator tmp(*this); tmp -= d; return tmp;}
    inline friend permutation_iterator operator+(const difference_type d, const permutation_iterator& z) {return z+d;}
    inline friend permutation_iterator operator-(const difference_type d, const permutation_iterator& z) {return z-d;}

    // Since operator== and operator!= are often used to terminate cycles,
    // defining them as follow prevents incrementing behind the end() of a container
    bool operator==(const permutation_iterator& rhs) const { return ((element_iterator == rhs.element_iterator) && (index_iterator == rhs.index_iterator)); }
    bool operator!=(const permutation_iterator& rhs) const { return !(this == rhs); }

  private:
    ElementIterator element_iterator;
    IndexIterator index_iterator;
};

} // namespace heimdall
} // namespace util
