#pragma once

#include <boost/format.hpp>
#include <string>

#include "gnuplot-iostream.h"

template <typename data, typename f>
void plot_with_tube(std::string title, const data &data, f ref_func,
                    double margin) {
  const std::string tubecolour = "#6699ff55";

  Gnuplot gp;
  gp << "set title '" << title << "'\n"
     << "plot '-' u 1:2:3 title 'acceptable margin: analytical ±"
     << boost::format("%.3f") % margin << "' w filledcu fs solid fc rgb '"
     << tubecolour
     << "', '-' u 1:2 "
        "title 'test result' w l\n";

  auto range = util::fmap(
      [&](auto x) {
        // with…
        const double t = x.first;
        const double analyt = ref_func(t);

        return std::make_tuple(t, analyt + margin, analyt - margin);
      },
      data);

  gp.send1d(range);
  gp.send1d(data);
  gp << "pause mouse key\nwhile (mouse_char ne 'q') { pause mouse "
        "key; }\n";
}
