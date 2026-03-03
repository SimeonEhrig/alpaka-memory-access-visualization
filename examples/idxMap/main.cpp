#include "vector_print.hpp"

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <sstream>
#include <string>

#ifdef DRAW_GRAPH_PNG
#    include "draw_data.hpp"
#endif


struct Kernel
{
    ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc, alpaka::concepts::IMdSpan auto out)
        const
    {
        alpaka::Vec<uint32_t, 1u> frame_number = acc[alpaka::frame::count].product();
        alpaka::concepts::CVector auto frame_extent = acc[alpaka::frame::extent];

        auto shared_input_data = alpaka::onAcc::declareSharedMdArray<int, alpaka::uniqueId()>(acc, frame_extent);

        // write the frame element id in the shared memory
        // if hardware thread process more then 1 element, it will lopp_counter * 100000 to its frame element id
        int loop_counter = 0;
        for(alpaka::concepts::Vector auto frame_elem : alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::linearThreadsInBlock,
                alpaka::onAcc::range::frameExtent))
        {
            shared_input_data[frame_elem] = frame_elem.product() + loop_counter * 100000;
            loop_counter += 1;
        }

        alpaka::onAcc::syncBlockThreads(acc);

        // use frame elem id for loading the data from the shared memory
        for(auto frame_index :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearBlocksInGrid, alpaka::IdxRange{frame_number}))
        {
            for(auto frame_elem : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::onAcc::range::frameExtent))
            {
                // calculate the global ID depending on the frame element ID, frame size and the frame ID
                auto const global_data_idx = frame_index * frame_extent + frame_elem;
                out[global_data_idx] = shared_input_data[frame_elem];
            }
        }
    }
};

int example(auto const deviceSpec, auto const exec)
{
    constexpr unsigned int data_size = 100u;
    constexpr unsigned int frame_extent = 32u;

    // #################################################
    // select device
    // #################################################

    // auto device_selector = alpaka::onHost::makeDeviceSelector(alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu);
    auto device_selector = alpaka::onHost::makeDeviceSelector(deviceSpec);

    auto num_devices = device_selector.getDeviceCount();
    std::cout << "Number of available Devices: " << num_devices << "\n";

    if(num_devices == 0)
    {
        return EXIT_FAILURE;
    }

    alpaka::onHost::Device acc_device = device_selector.makeDevice(0);
    std::cout << "Device 0: " << acc_device.getName() << "\n";

    alpaka::onHost::Queue acc_queue = acc_device.makeQueue(alpaka::queueKind::blocking);

    // #################################################
    // allocate memory and apply transformation
    // #################################################

    auto extents = alpaka::CVec<int, data_size>{};

    auto acc_out = alpaka::onHost::alloc<int>(acc_device, extents);

    unsigned int num_frames = alpaka::divCeil(data_size, frame_extent);

    std::cout << "Data size: " << data_size << "\n";
    std::cout << "Use " << num_frames << " FrameSpecs with a size of " << frame_extent << "\n";
    std::cout << "Use executor: " << alpaka::onHost::getName(exec) << "\n";

    // define a specific frameSpec size and number of frameSpecs
    auto frame_spec = alpaka::onHost::FrameSpec{num_frames, alpaka::CVec<uint32_t, frame_extent>{}};
    acc_queue.enqueue(exec, frame_spec, alpaka::KernelBundle{Kernel{}, acc_out});

    auto host_out = alpaka::onHost::allocHost<int>(data_size);
    alpaka::onHost::memcpy(acc_queue, host_out, acc_out);

    // #################################################
    // verify result
    // #################################################

    if(data_size <= 100)
    {
        std::cout << "\n" << dataSourceToStr(host_out) << "\n\n";
    }
    std::cout << dataSourceToStrSparse(host_out) << "\n";

#ifdef DRAW_GRAPH_PNG
    drawGraphPNG(host_out, alpaka::onHost::getName(exec));
#endif

    return 0;
}

int main(int argc, char** argv)
{
    return alpaka::onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors));
}
