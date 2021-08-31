#pragma once

#include <boost/compute/detail/meta_kernel.hpp>
#include <boost/compute/iterator/discard_iterator.hpp>

class discard_iterator_wrapper;

template<class IndexExpr>
struct discard_iterator_wrapper_index_expr
{
    typedef void result_type;

    discard_iterator_wrapper_index_expr(const IndexExpr &expr)
        : m_expr(expr)
    {
    }

    IndexExpr m_expr;
};

namespace boost::compute {
namespace detail {

template<class IndexExpr>
inline meta_kernel& operator<<(meta_kernel &kernel,
                               const ::discard_iterator_wrapper_index_expr<IndexExpr> &expr)
{
    (void) expr;
    kernel << "//";

    return kernel;
}

} // namespace boost::compute::detail

/// internal_ (is_device_iterator specialization for discard_iterator_wrapper)
template<>
struct is_device_iterator<discard_iterator_wrapper> : boost::true_type {};

} // namespace boost::compute

/// a hack: use "//" to comment out unused variable/expression
class discard_iterator_wrapper : public boost::compute::discard_iterator {
public:
    using boost::compute::discard_iterator::discard_iterator;
    discard_iterator_wrapper(boost::compute::discard_iterator iter) : boost::compute::discard_iterator::discard_iterator(iter) {}

    /// \internal_
    template<class Expr>
    discard_iterator_wrapper_index_expr<Expr>
    operator[](const Expr &expr) const
    {
        return discard_iterator_wrapper_index_expr<Expr>(expr);
    }
};