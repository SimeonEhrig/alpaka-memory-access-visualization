#include "reduce_helper.hpp"
#include "reduce_kernel.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

using TestBackends
    = std::decay_t<decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("test reduce example", "[example]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];

    INFO("Device spec: " << alpaka::onHost::getName(deviceSpec) << ", Executor: " << alpaka::onHost::getName(exec));

    std::size_t const n = GENERATE(192, 253, 10457);
    std::size_t const frame_extent = GENERATE(2, 8, 16, 32);
    alpaka::onHost::FrameSpec frame_spec = get_reduce_frame_spec(n, frame_extent);
    std::size_t const num_frames = frame_spec.getNumFrames().product();

    INFO(
        "Problem size: " << n << "\n  Frame extents: " << frame_extent << "\n  Number frames: " << num_frames << "\n");

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(devSelector.getDeviceCount() > 0);

    auto device = devSelector.makeDevice(0);
    INFO("Use device: " << device.getName());

    auto queue = device.makeQueue();

    alpaka::concepts::IBuffer<int> auto host_input = alpaka::onHost::allocHost<int>(n);
    alpaka::concepts::IBuffer<int> auto device_input = alpaka::onHost::alloc<int>(device, n);
    alpaka::concepts::IBuffer<int> auto host_output = alpaka::onHost::allocHost<int>(num_frames);
    alpaka::concepts::IBuffer<int> auto device_output = alpaka::onHost::alloc<int>(device, num_frames);
    std::vector<int> host_output_vec(num_frames, 0);


    // initialize data
    for(auto i = 0; i < n; ++i)
    {
        host_input[i] = i;
    }

    alpaka::onHost::memcpy(queue, device_input, host_input);
    queue.enqueue(frame_spec, alpaka::KernelBundle{ReduceKernel{}, device_input, device_output});
    alpaka::onHost::memcpy(queue, host_output_vec, device_output);
    alpaka::onHost::wait(queue);

    SECTION("Check partial sums")
    {
        std::vector<int> expected_partial_sums = calculate_partial_sums(n, frame_extent);

        REQUIRE(host_output.getExtents().product() == expected_partial_sums.size());
        REQUIRE_THAT(host_output_vec, Catch::Matchers::Equals(expected_partial_sums));
    }

    SECTION("Check complete sum")
    {
        // reduce the partial block sums to a single value
        int calculated_sum = 0;
        for(auto i = 0; i < num_frames; ++i)
        {
            calculated_sum += host_output_vec[i];
        }

        // calculate the expected result with the gaussian sum form
        int expected_sum = ((n - 1) * ((n - 1) + 1)) / 2;

        REQUIRE(calculated_sum == expected_sum);
    }
}
