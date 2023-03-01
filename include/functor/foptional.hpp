// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65

#pragma once

#include "functor.hpp"
#include <algorithm>
#include <functional>
#include <optional>
#include <type_traits>

// fmap : (A → B) → F<A> → F<B>
//
// Accepts optional value by reference, and then dereferences on
// pass to invoke.
template <typename A, typename F>
auto fmap(F f, const std::optional<A> &oa) {
  if (oa)
    return std::make_optional(std::invoke(f, *oa));

  return std::optional<std::invoke_result_t<F, A>>{};
}

// fmap : (A → B) → F<A> → F<B>
//
// Takes optional by universal reference and moves from the
// dereferenced optional.
template <typename A, typename F>
auto fmap(F f, std::optional<A> &&oa) {
  if (oa)
    return std::make_optional(std::invoke(f, std::move(*oa)));

  return std::optional<std::invoke_result_t<F, A>>{};
}
