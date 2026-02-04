#include "reduce_helper.hpp"
#include "reduce_kernel.hpp"

#include <alpaka/alpaka.hpp>

#include <iostream>

int main()
{
    constexpr std::size_t n = 192;
    constexpr std::size_t frame_extent = 32;
    alpaka::onHost::FrameSpec const frame_spec = get_reduce_frame_spec(n, frame_extent);
    std::size_t const num_frames = frame_spec.m_numFrames.product();

    std::cout << "Problem size: " << n << "\nFrame extents: " << frame_extent << "\nNumber frames: " << num_frames
              << "\n";

    auto devSelector = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
    if(devSelector.getDeviceCount() == 0)
    {
        std::cerr << "No device available! Exit application.\n";
        return 1;
    }

    auto device = devSelector.makeDevice(0);
    std::cout << "Use device: " << device.getName() << "\n";
    auto queue = device.makeQueue();

    alpaka::concepts::IBuffer<int> auto input = alpaka::onHost::allocUnified<int>(device, n);
    alpaka::concepts::IBuffer<int> auto output = alpaka::onHost::allocUnified<int>(device, num_frames);

    // initialize data
    for(auto i = 0; i < n; ++i)
    {
        input[i] = i;
    }


    queue.enqueue(frame_spec, alpaka::KernelBundle{ReduceKernel{}, input, output});
    alpaka::onHost::wait(queue);

    // reduce the partial block sums to a single value
    int calculated_sum = 0;
    for(auto i = 0; i < num_frames; ++i)
    {
        calculated_sum += output[i];
    }

    // calculate the expected result with the gaussian sum form
    int expected_sum = ((n - 1) * ((n - 1) + 1)) / 2;

    if(calculated_sum == expected_sum)
    {
        std::cout << "Result is correct: " << calculated_sum << "\n";
        return 0;
    }
    else
    {
        std::cout << "Result is wrong\ncalculated: " << calculated_sum << "\nexpected:   " << expected_sum << "\n";
        std::cout << "input: ";
        for(auto e : input)
        {
            std::cout << e << " ";
        }
        std::cout << "\n";
        std::cout << "expected output: ";
        for(auto e : calculate_partial_sums(n, frame_extent))
        {
            std::cout << e << " ";
        }
        std::cout << "\n";

        std::cout << "         output: ";
        for(auto e : output)
        {
            std::cout << e << " ";
        }
        std::cout << "\n";
        return 0;
    }
}
