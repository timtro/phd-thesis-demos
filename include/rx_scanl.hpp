// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#pragma once

#include "Cpp-BiCCC.hpp"

#include <rxcpp/rx.hpp>

namespace rx {
  using namespace rxcpp;
  using namespace rxcpp::operators;
  using namespace rxcpp::sources;
  using namespace rxcpp::util;
}  // namespace rx 

template <typename I, typename S>
auto rx_scanl(S s0, Hom<Doms<S, I>, S> f)
    -> Hom<rx::observable<I>, rx::observable<S>> {
  return [s0, f](rx::observable<I> obs) {
    return obs.scan(s0, f).start_with(s0);
  };
}

