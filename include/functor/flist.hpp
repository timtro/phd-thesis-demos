// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65

#pragma once

#include "functor.hpp"

#include <algorithm>
#include <array>
#include <experimental/type_traits>
#include <functional>
#include <iterator>
#include <type_traits>

using std::begin;
using std::cbegin;
using std::cend;
using std::end;

using std::experimental::is_detected_v;

namespace tf {

  namespace dtl {
    template <class T>
    using has_reserve_t = decltype(std::declval<T &>().reserve(
        std::declval<size_t>()));

    template <typename T>
    constexpr bool has_reserve_v =
        is_detected_v<has_reserve_t, T>;
  } // namespace dtl

  // fmap : (A → B) → F<A> → F<B>
  //
  // This version specifies on a collection functor, F,  which
  // can be constructed with a typename (F<typename>) or list of
  // typenames. This includes std::vector and std::list. If F has
  // a reserve(·) member like std::vector does, it is called to
  // avoid needless allocations.
  template <template <typename...> typename Functor, typename A,
      typename... FCtorArgs, typename F>
  auto fmap(F f, const Functor<A, FCtorArgs...> &as) {
    Functor<std::invoke_result_t<F, A>> bs;

    if constexpr (dtl::has_reserve_v<decltype(bs)>)
      bs.reserve(as.size());

    std::transform(cbegin(as), cend(as), std::back_inserter(bs),
        std::forward<F>(f));
    return bs;
  }

  // fmap : (A → B) → F<A> → F<B>
  //
  // This version is specific to std::array, which can't be
  // constructed as std::array<typename...>, because it needs a
  // size_t. This makes it a dependent type, which requires a
  // little more care than a regular functor.
  template <typename F, typename A, size_t N>
  auto fmap(F f, const std::array<A, N> &as) {
    std::array<std::invoke_result_t<F, A>, N> bs;

    std::transform(cbegin(as), cend(as), begin(bs), f);
    return bs;
  }

} // namespace tf
