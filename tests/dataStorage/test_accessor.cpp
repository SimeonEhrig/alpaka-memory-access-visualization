#include <alpaka/alpaka.hpp>
#include <alpaka/mav/mav.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <limits>
#include <vector>

struct ReduceKernel
{
    ALPAKA_FN_ACC void operator()(
        alpaka::onAcc::concepts::Acc auto const& acc,
        alpaka::concepts::IDataSource auto raw_input,
        alpaka::concepts::IMdSpan auto output,
        alpaka::concepts::IMdSpan auto access_data

    ) const
    {
        alpaka::mav::onAcc::MdSpan input{acc, raw_input, access_data};

        using Vec = typename decltype(acc[alpaka::frame::count])::UniVec;
        using VecType = typename Vec::type;

        Vec const frame_number = acc[alpaka::frame::count];
        Vec const frame_extent = acc[alpaka::frame::extent];
        Vec const thread_id = acc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);

        // iterate over all frames
        for(alpaka::concepts::Dim<1> auto frame_index :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearBlocksInGrid, alpaka::IdxRange{frame_number}))
        {
            // the element offset defines, which element is added with each other element
            // e.g. input[0] += input[n/2], input[1] += input[(n/2) + 1] ...
            for(alpaka::concepts::Dim<1> auto elem_offset = frame_extent; elem_offset > Vec{0}; elem_offset /= Vec{2})
            {
                // iterate over all frame elements and therefore over all left elements
                for(alpaka::concepts::Dim<1> auto frame_elem : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::linearThreadsInBlock,
                        alpaka::IdxRange{frame_extent}))
                {
                    if(frame_elem < elem_offset)
                    {
                        Vec left_elem = frame_index * frame_extent * VecType{2u} + frame_elem;
                        Vec right_elem = left_elem + elem_offset;

                        // We must check whether the right element is within the total number of elements in case the
                        // number of elements cannot be divided by frame_extent*2 without a remainder.
                        if(right_elem < input.getExtents())
                        {
                            input[left_elem] += input[right_elem];
                        }
                    }
                }
                alpaka::onAcc::syncBlockThreads(acc);
            }
            // select only local thread to store result in output
            if(thread_id == Vec{0})
            {
                output[frame_index] = input[frame_index * frame_extent * VecType{2u}];
            }
        }
    }
};

using TestBackends
    = std::decay_t<decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE(
    "test if acccessor count the correct number of memory access",
    "[onAcc][accessor]",
    TestBackends)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];

    INFO("Device spec: " << alpaka::onHost::getName(deviceSpec) << ", Executor: " << alpaka::onHost::getName(exec));

    constexpr std::size_t n = 8;
    constexpr std::size_t frame_extents = 4;
    auto frame_spec = alpaka::onHost::FrameSpec(alpaka::divCeil(n, frame_extents * 2), frame_extents);
    std::size_t num_frames = frame_spec.m_numFrames.product();

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(devSelector.getDeviceCount() > 0);

    auto device = devSelector.makeDevice(0);
    INFO("Use device: " << device.getName());

    auto queue = device.makeQueue();

    alpaka::concepts::IBuffer<int> auto input = alpaka::onHost::allocUnified<int>(queue, n);
    alpaka::concepts::IBuffer<int> auto output = alpaka::onHost::allocUnified<int>(queue, num_frames);
    alpaka::onHost::iota(queue, int{0}, input);

    using Extents = ALPAKA_TYPEOF(input.getExtents());
    using AccessInfo = alpaka::mav::AccessInfo<Extents, Extents>;
    alpaka::concepts::IBuffer<AccessInfo> auto access_data = alpaka::onHost::allocUnified<AccessInfo>(queue, n);
    alpaka::onHost::fill(
        queue,
        access_data,
        AccessInfo{Extents::fill(0), Extents::fill(0), Extents::fill(0), 0, true});

    queue.enqueue(exec, frame_spec, alpaka::KernelBundle{ReduceKernel{}, input, output, access_data});
    alpaka::onHost::wait(queue);

    int calculated_sum = 0;
    for(auto i = 0; i < num_frames; ++i)
    {
        calculated_sum += output[i];
    }

    int expected_sum = ((n - 1) * ((n - 1) + 1)) / 2;
    REQUIRE(calculated_sum == expected_sum);

    // number of memory access per position
    std::vector<std::size_t> expected_memory_accesses = {7, 5, 3, 3, 1, 1, 1, 1};
    REQUIRE(expected_memory_accesses.size() == n);

    std::vector<std::size_t> counted_memory_accesses(
        expected_memory_accesses.size(),
        std::numeric_limits<std::size_t>::max());

    for(auto i = 0; i < counted_memory_accesses.size(); ++i)
    {
        counted_memory_accesses[i] = access_data[i].access_counter;
    }

    REQUIRE_THAT(counted_memory_accesses, Catch::Matchers::Equals(expected_memory_accesses));
}
