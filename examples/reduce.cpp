#include <alpaka/alpaka.hpp>

#include <iostream>

struct ReduceKernel
{
    ALPAKA_FN_ACC void operator()(
        alpaka::onAcc::concepts::Acc auto const& acc,
        alpaka::concepts::IDataSource auto input,
        alpaka::concepts::IMdSpan auto output) const
    {
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

    for(std::size_t i = 0; ((i + 1) * frame_extent * 2) < n; ++i)
    {
        partial_sums[i] = gauss_sum(((i + 1) * frame_extent * 2) - 1) - gauss_sum((i * frame_extent * 2) - 1);
    }
    if(frame_extent * 2 * num_frames > n)
    {
        partial_sums[num_frames - 1] = gauss_sum(n - 1) - gauss_sum(((num_frames - 1) * frame_extent * 2) - 1);
    }

    return partial_sums;
}

int main()
{
    constexpr std::size_t n = 194;
    constexpr std::size_t frame_extent = 2;
    constexpr std::size_t num_frames = alpaka::divCeil(n, frame_extent * 2);

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

    auto const frame_spec = alpaka::onHost::FrameSpec(num_frames, frame_extent);
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
