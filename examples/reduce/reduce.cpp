// set to 0 to disable mav support
#define USE_MAV 1
#include "reduce_helper.hpp"
#include "reduce_kernel.hpp"

#include <alpaka/alpaka.hpp>

#include <iostream>

#if USE_MAV == 1
#    include <alpaka/mav/mav.hpp>

#    include <filesystem>
#    include <fstream>
#endif

int example(auto const deviceSpec, auto const exec, int argc, char** argv)
{
    constexpr std::size_t n = 192;
    constexpr std::size_t frame_extent = 32;
    alpaka::onHost::FrameSpec const frame_spec = get_reduce_frame_spec(n, frame_extent);
    std::size_t const num_frames = frame_spec.m_numFrames.product();

    std::cout << "Problem size: " << n << "\nFrame extents: " << frame_extent << "\nNumber frames: " << num_frames
              << "\n";

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
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

#if USE_MAV == 1
    using Extents = ALPAKA_TYPEOF(input.getExtents());
    using AccessInfo = alpaka::mav::AccessInfo<Extents, Extents>;
    alpaka::concepts::IBuffer<AccessInfo> auto access_data = alpaka::onHost::allocUnified<AccessInfo>(device, n);
    alpaka::onHost::fill(
        queue,
        access_data,
        AccessInfo{Extents::fill(0), Extents::fill(0), Extents::fill(0), 0, true});
#endif

    std::cout << "Use executor: " << alpaka::onHost::getName(exec) << "\n";
    queue.enqueue(
        exec,
        frame_spec,
        alpaka::KernelBundle{
            ReduceKernel{},
            input,
            output
#if USE_MAV == 1
            ,
            access_data
#endif
        });
    alpaka::onHost::wait(queue);

#if USE_MAV == 1
    auto access_json = alpaka::mav::getJSONFromData(
        alpaka::mav::onHost::MdSpan("reduce" + alpaka::onHost::getName(exec), input, access_data),
        frame_spec);
    auto exe_dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();
    auto access_data_path = exe_dir / ("reduce_" + alpaka::onHost::getName(exec) + ".json");
    std::cout << "write access data to: " << access_data_path << "\n";
    std::ofstream file(access_data_path);
    if(file.is_open())
    {
        file << access_json;
        file.close();
    }
    else
    {
        std::cerr << "ERROR: Could not write to " << access_data_path << "\n";
    }
#endif

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

int main(int argc, char** argv)
{
    return alpaka::onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec], argc, argv); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors));
}
