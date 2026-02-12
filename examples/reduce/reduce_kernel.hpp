#pragma once

#include <alpaka/alpaka.hpp>
#if USE_MAV == 1
#    include <alpaka/mav/mav.hpp>
#endif

struct ReduceKernel
{
    ALPAKA_FN_ACC void operator()(
        alpaka::onAcc::concepts::Acc auto const& acc,
        alpaka::concepts::IDataSource auto raw_input,
        alpaka::concepts::IMdSpan auto output
#if USE_MAV == 1
        ,
        alpaka::concepts::IMdSpan auto access_data
#endif
    ) const
    {
#if USE_MAV == 1
        alpaka::mav::onAcc::MdSpan input{acc, raw_input, access_data};
#else
        auto& input = raw_input;
#endif

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
