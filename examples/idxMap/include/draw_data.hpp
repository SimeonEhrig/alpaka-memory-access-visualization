#pragma once

#include <alpaka/alpaka.hpp>

#include <matplot/matplot.h>

#include <string>

void drawGraphPNG(alpaka::concepts::IDataSource auto data, std::string const filename)
{
    static_assert(data.dim() == 1);

    std::vector<int> y(data.getExtents()[0]);

    for(auto i = 0; i < data.getExtents()[0]; ++i)
    {
        y[i] = data[i];
    }
    matplot::figure(true);
    matplot::stairs(y, "blue");

    matplot::save(filename, "png");
}
