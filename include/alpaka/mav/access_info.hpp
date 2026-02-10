#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka::mav
{
    template<alpaka::concepts::Vector T_DataIndex, alpaka::concepts::Vector T_ThreadIndex>
    struct AccessInfo
    {
        T_DataIndex data_index;
        T_ThreadIndex local_thread_index;
        T_ThreadIndex block_index;
        std::size_t access_counter = 0;
        // false == read
        // true == write
        bool read_write;
    };
 template<alpaka::concepts::Vector T_DataIndex, alpaka::concepts::Vector T_ThreadIndex>
    std::ostream& operator<<(std::ostream& s, AccessInfo<T_DataIndex, T_ThreadIndex> const& ai)
    {
        return s << "AccessInfo{ index=" << ai.data_index << ", local thread idx=" << ai.local_thread_index
                 << ", block idx=" << ai.block_index << ", num access=" << ai.access_counter << ", "
                 << (ai.read_write ? "write" : "read") << " }";
    }
} // namespace alpaka::mav
