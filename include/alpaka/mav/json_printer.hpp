#pragma once
#include "mdspan.hpp"

namespace alpaka::mav
{
    std::string getJSONFromData(
        alpaka::mav::onHost::concepts::MdSpan auto const& data,
        alpaka::onHost::concepts::FrameSpec auto const& frameSpec,
        alpaka::concepts::IDataSource auto const& counter_data,
        std::integral auto const max_numbers_logs)
    {
        static_assert(ALPAKA_TYPEOF(data)::dim() <= 3);
        std::stringstream jsStream;

        // Create a JSON object which describes all the available data
        // We will have one key GlobalSettings, one key MemoryRegions, and one key MemoryAccessLogs

        // Start the JSON object
        jsStream << "{";

        // Add the GlobalSettings key
        jsStream << "\"GlobalSettings\": {";

        // Store all settings

        alpaka::Vec<typename ALPAKA_TYPEOF(data.getExtents())::type, 3> grid_dim{1, 1, 1};
        for(auto i = 0; i < ALPAKA_TYPEOF(data.getExtents())::dim(); ++i)
        {
            grid_dim[i] = data.getExtents()[i];
        }
        // First store the grid dimensions
        jsStream << "\"GridDimensions\": {";
        jsStream << "\"x\": " << grid_dim.x() << ",";
        jsStream << "\"y\": " << grid_dim.y() << ",";
        jsStream << "\"z\": " << grid_dim.z();
        jsStream << "},";

        alpaka::Vec<typename ALPAKA_TYPEOF(frameSpec)::index_type, 3> block_dim{1, 1, 1};
        for(auto i = 0; i < ALPAKA_TYPEOF(frameSpec.getFrameExtents())::dim(); ++i)
        {
            block_dim[i] = frameSpec.getFrameExtents()[i];
        }
        // Then store the block dimensions
        jsStream << "\"BlockDimensions\": {";
        jsStream << "\"x\": " << block_dim.x() << ",";
        jsStream << "\"y\": " << block_dim.y() << ",";
        jsStream << "\"z\": " << block_dim.z();
        jsStream << "},";

        // TODO: clarify what is the warpsize in alpaka 3
        // Then store the warp size
        jsStream << "\"WarpSize\": " << frameSpec.getFrameExtents().x();

        // Store the original size
        jsStream << ",\"OriginalSize\": " << data.get_access_mdspan().getExtents().product();

        // Store the current size
        jsStream << ",\"CurrentSize\": " << data.get_access_mdspan().getExtents().product();

        // Close the GlobalSettings object
        jsStream << "},";

        // Add the MemoryRegions key
        jsStream << "\"MemoryRegions\": [";

        // Store the start address, number of elements and the size of a single element
        jsStream << "{";
        jsStream << "\"StartAddress\": \"" << std::hex << &(*data.begin()) << "\",";
        // As the memory region is already a pointer of the correct data type, we do not actually need to multiply
        // the size by the size of the data type

        using IndexVecType = ALPAKA_TYPEOF(data.getExtents());
        // the web viewer requires, that the end address is the first byte, which is not part of the memory
        jsStream << "\"EndAddress\": \"" << std::hex << data.ptr(data.getExtents()) << "\",";
        jsStream << "\"NumberOfElements\": " << std::dec << data.getExtents().product() << ",";
        jsStream << "\"SizeOfSingleElement\": " << sizeof(typename ALPAKA_TYPEOF(data)::value_type) << ",";
        jsStream << "\"Name\": \"" << data.get_name() << "\"";
        jsStream << "}";

        // Close the MemoryRegions array
        jsStream << "],";

        // Add the MemoryAccessLogs key

        jsStream << "\"MemoryAccessLogs\": [";

        // TODO: the loop works only with 1D data
        static_assert(ALPAKA_TYPEOF(data)::dim() == 1);
        // Loop through all memory access logs and store them
        // To save on storage space and parsing time (as this is the biggest part of the data) we store an array with
        // the data entries directly instead of using an object with names for each entry
        for(alpaka::concepts::Vector auto idx : alpaka::IdxRange(counter_data.getExtents()))
        {
            using IdxType = typename ALPAKA_TYPEOF(data.getExtents())::type;
            for(IdxType log_number = 0; log_number < counter_data[idx]; ++log_number)
            {
                if(log_number >= max_numbers_logs)
                {
                    std::cerr
                        << "WARNING: idx=" << idx << " counted " << counter_data[idx] << " memory accesses. But only "
                        << max_numbers_logs
                        << " can be stored. Increase the number of logs per element to store all access events.\n";
                    break;
                }
                auto info = data.get_access_mdspan()[idx * max_numbers_logs + log_number];

                jsStream << "[";
                jsStream << "\"" << std::hex << data.ptr(idx) << "\",";
                jsStream << std::dec << info.block_index.x() << ",";
                jsStream << info.local_thread_index.x() << ",";
                jsStream << (info.read_write ? "true" : "false");
                jsStream << "]";

                jsStream << ",";
            }
        }

        // decrease the write cursor position by 1 to overwrite the last comma with the parentheses ]
        jsStream.seekp(-1, std::ios_base::end);
        // Close the MemoryAccessLogs array
        jsStream << "]";

        // Close the JSON object
        jsStream << "}";
        return jsStream.str();
    }
} // namespace alpaka::mav
