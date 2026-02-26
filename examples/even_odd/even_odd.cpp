#include <alpaka/alpaka.hpp>
#include <alpaka/mav/mav.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>

struct EvenOddKernel
{
    ALPAKA_FN_ACC void operator()(
        alpaka::onAcc::concepts::Acc auto const& acc,
        alpaka::concepts::IMdSpan auto data,
        alpaka::concepts::IMdSpan auto access_data,
        alpaka::concepts::IMdSpan auto counter_data) const
    {
        alpaka::mav::onAcc::MdSpan mav_data{acc, data, access_data, counter_data};
        for(alpaka::concepts::Dim<1> auto idx : alpaka::onAcc::makeIdxMap(
                acc,
                alpaka::onAcc::worker::linearThreadsInGrid,
                alpaka::IdxRange{data.getExtents()}))
        {
            if(idx.product() % 2 == 0)
            {
                int k = mav_data[idx];
            }
            else
            {
                mav_data[idx] = 1;
            }
        }
    }
};

int example(auto const deviceSpec, auto const exec, int argc, char** argv)
{
    constexpr std::size_t n = 40;
    constexpr std::size_t frame_extent = 4;
    constexpr std::size_t num_logs_per_element = 10;

    alpaka::onHost::FrameSpec const frame_spec
        = alpaka::onHost::FrameSpec(alpaka::divCeil(n, frame_extent), frame_extent);
    std::size_t const num_frames = frame_spec.getNumFrames().product();

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

    alpaka::concepts::IBuffer<int> auto data = alpaka::onHost::allocUnified<int>(queue, n);
    using Extents = ALPAKA_TYPEOF(data.getExtents());
    using ExtentsType = typename Extents::type;
    using AccessInfo = alpaka::mav::AccessInfo<Extents>;
    alpaka::concepts::IBuffer<AccessInfo> auto access_data
        = alpaka::onHost::allocUnified<AccessInfo>(queue, n * num_logs_per_element);
    alpaka::concepts::IBuffer<ExtentsType> auto counter_data = alpaka::onHost::allocUnified<ExtentsType>(queue, n);

    alpaka::onHost::fill(queue, data, 0);
    alpaka::onHost::fill(queue, access_data, AccessInfo{Extents::fill(0), Extents::fill(0), true});
    alpaka::onHost::fill(queue, counter_data, ExtentsType{0});

    std::cout << "Use executor: " << alpaka::onHost::getName(exec) << "\n";
    queue.enqueue(exec, frame_spec, alpaka::KernelBundle{EvenOddKernel{}, data, access_data, counter_data});
    alpaka::onHost::wait(queue);

    std::cout << "data: ";
    for(auto e : data)
    {
        std::cout << e << " ";
    }
    std::cout << "\n";


    auto access_json = alpaka::mav::getJSONFromData(
        alpaka::mav::onHost::MdSpan("even_odd_" + alpaka::onHost::getName(exec), data, access_data),
        frame_spec,
        counter_data,
        num_logs_per_element);
    auto exe_dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();
    auto access_data_path = exe_dir / ("even_odd_" + alpaka::onHost::getName(exec) + ".json");
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
        return 1;
    }

    return 0;
}

int main(int argc, char** argv)
{
    return alpaka::onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec], argc, argv); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors));
}
