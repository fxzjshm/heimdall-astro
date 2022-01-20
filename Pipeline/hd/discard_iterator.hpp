#pragma once

#include <type_traits>

/**
 *  @brief  Discard output from some algorithms. Modfied from ostream_iterator and ZipIter
 *
 *  @tparam  T  The type to write to the ostream.
*/
class discard_iterator {
  public:
    using difference_type = ptrdiff_t;

    /// Construct from nothing
    discard_iterator() {}

    /// Writes @a value to nothing
    template<typename T>
    discard_iterator& operator=(const T& value) {
      (void) value;
      return *this;
    }

    discard_iterator& operator*() { return *this; }
    discard_iterator& operator++() { return *this; }
    discard_iterator& operator++(int) { return *this; }
    discard_iterator& operator+=(const difference_type d) { (void) d; return *this; }
    discard_iterator& operator-=(const difference_type d) { return operator+=(-d); }
    discard_iterator operator+(const difference_type d) const {discard_iterator tmp(*this); tmp += d; return tmp;}
    discard_iterator operator-(const difference_type d) const {discard_iterator tmp(*this); tmp -= d; return tmp;}

    template<class Expr>
    discard_iterator&
    operator[](Expr &expr)
    {
      (void) expr;
      return *this;
    }
};