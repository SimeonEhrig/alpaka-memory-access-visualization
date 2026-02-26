#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka::mav
{
    template<alpaka::concepts::Vector T_ThreadIndex>
    struct AccessInfo
    {
        T_ThreadIndex local_thread_index;
        T_ThreadIndex block_index;
        // true == read
        // false == write
        bool read_write;
    };

    template<alpaka::concepts::Vector T_ThreadIndex>
    std::ostream& operator<<(std::ostream& s, AccessInfo<T_ThreadIndex> const& ai)
    {
        return s << "AccessInfo{ local thread idx=" << ai.local_thread_index << ", block idx=" << ai.block_index
                 << ", " << (ai.read_write ? "read" : "write") << " }";
    }
} // namespace alpaka::mav
