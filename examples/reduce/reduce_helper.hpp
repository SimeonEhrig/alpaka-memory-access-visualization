#pragma once

#include <alpaka/alpaka.hpp>

/// Generates frameSpec sepcific for the reduce kernel
/// @param n Total number of elements.
/// @param frame_extent Size of the frameSpec
/// @return frameSpec specific for the reduce kernel
[[nodiscard]] inline auto get_reduce_frame_spec(std::size_t const n, std::size_t const frame_extent)
{
    return alpaka::onHost::FrameSpec(alpaka::divCeil(n, frame_extent * 2), frame_extent);
}

/// Calculates the partial sum for the block wise reduction algorithm. Uses the gaussian sum form. Therefore, it only
/// works if the input data for real algorithm is initialized with 0 until n -1.
///
/// @param n Total number of elements.
/// @param frame_extent Size of the frameSpec
/// @return Returns an vector of partial sums. The number of elements is equal to the number of frameSpecs.
std::vector<int> calculate_partial_sums(std::size_t const n, std::size_t const frame_extent)
{
    std::size_t const num_frames = alpaka::divCeil(n, frame_extent * 2);
    std::vector<int> partial_sums(num_frames, 0);

    auto gauss_sum = [](int const n) { return (n * (n + 1)) / 2; };

    for(std::size_t i = 0; ((i + 1) * frame_extent * 2) <= n; ++i)
    {
        partial_sums[i] = gauss_sum(((i + 1) * frame_extent * 2) - 1) - gauss_sum((i * frame_extent * 2) - 1);
    }
    if(frame_extent * 2 * num_frames > n)
    {
        partial_sums[num_frames - 1] = gauss_sum(n - 1) - gauss_sum(((num_frames - 1) * frame_extent * 2) - 1);
    }

    return partial_sums;
}
