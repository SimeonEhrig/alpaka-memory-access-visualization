#include <alpaka/alpaka.hpp>
#include <alpaka/mav/mav.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>


using TestBackends
    = std::decay_t<decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors))>;

struct Kernel
{
    ALPAKA_FN_ACC void operator()(
        alpaka::onAcc::concepts::Acc auto const& acc,
        alpaka::concepts::IMdSpan auto data,
        alpaka::concepts::IMdSpan auto access_data) const
    {
        alpaka::mav::onAcc::MdSpan mav_span(acc, data, access_data);
        for(auto frame_index : alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::linearBlocksInGrid,
                alpaka::IdxRange{acc[alpaka::frame::count]}))
        {
            for(auto frame_elem : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInBlock,
                    alpaka::IdxRange{acc[alpaka::frame::extent]}))
            {
                auto global_thread_frame_id = frame_index * acc[alpaka::frame::extent] + frame_elem;
                if(global_thread_frame_id < data.getExtents())
                {
                    mav_span[global_thread_frame_id] = global_thread_frame_id.product();
                }
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("test manually constructed onAcc::mdspan accessor", "[onAcc][mdspan]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];

    INFO("Device spec: " << alpaka::onHost::getName(deviceSpec) << ", Executor: " << alpaka::onHost::getName(exec));

    std::size_t const n = GENERATE(192, 253, 10457);
    std::size_t const frame_extents = GENERATE(2, 10, 16, 32);

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(devSelector.getDeviceCount() > 0);

    auto device = devSelector.makeDevice(0);
    INFO("Use device: " << device.getName());

    auto queue = device.makeQueue();

    alpaka::concepts::IBuffer<int> auto device_data = alpaka::onHost::alloc<int>(device, n);
    using Extents = ALPAKA_TYPEOF(device_data.getExtents());
    using AccessInfo = alpaka::mav::AccessInfo<Extents, Extents>;
    alpaka::concepts::IBuffer<AccessInfo> auto device_access_data = alpaka::onHost::alloc<AccessInfo>(device, n);

    alpaka::onHost::fill(queue, device_data, 1);
    alpaka::onHost::fill(
        queue,
        device_access_data,
        AccessInfo{Extents::fill(0), Extents::fill(0), Extents::fill(0), 0, true});

    auto frame_spec = alpaka::onHost::FrameSpec(alpaka::divCeil(n, frame_extents), frame_extents);
    queue.enqueue(exec, frame_spec, alpaka::KernelBundle(Kernel{}, device_data, device_access_data));

    alpaka::concepts::IBuffer<int> auto host_data = alpaka::onHost::allocHost<int>(n);
    alpaka::concepts::IBuffer<AccessInfo> auto host_access_data = alpaka::onHost::allocHost<AccessInfo>(n);

    alpaka::onHost::memcpy(queue, host_data, device_data);
    alpaka::onHost::memcpy(queue, host_access_data, device_access_data);

    alpaka::onHost::wait(queue);

    for(auto i = 0u; i < n; ++i)
    {
        REQUIRE(host_data[i] == i);
    }

    for(auto i = 0u; i < n; ++i)
    {
        REQUIRE(host_access_data[i].access_counter == 1);
    }
}
