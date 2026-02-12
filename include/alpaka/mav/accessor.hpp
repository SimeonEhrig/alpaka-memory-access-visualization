#pragma once

#include <iostream>

namespace alpaka::mav::details
{
    template<typename TData>
    class AccessProxy
    {
        TData* const m_ptr;
        std::size_t& m_access_counter;
        bool& m_read_write;

    public:
        constexpr AccessProxy(TData* const ptr, std::size_t& access_counter, bool& read_write)
            : m_ptr(ptr)
            , m_access_counter(access_counter)
            , m_read_write(read_write)
        {
        }

        constexpr AccessProxy& operator=(TData const& value)
        {
            *m_ptr = value;
            m_read_write = false;
            m_access_counter += 1;
            return *this;
        }

        constexpr operator TData() const
        {
            m_read_write = true;
            m_access_counter += 1;
            return *m_ptr;
        }

        // It's a read/write. Therefore add 2 accesses.
#define ALPAKA_MAV_ACCESSOR_ASSIGN_OP(op)                                                                             \
    constexpr AccessProxy& operator op(TData const& value)                                                            \
    {                                                                                                                 \
        m_read_write = false;                                                                                         \
        *m_ptr op value;                                                                                              \
        m_access_counter += 2;                                                                                        \
        return *this;                                                                                                 \
    }

        ALPAKA_MAV_ACCESSOR_ASSIGN_OP(+=)
        ALPAKA_MAV_ACCESSOR_ASSIGN_OP(-=)
        ALPAKA_MAV_ACCESSOR_ASSIGN_OP(/=)
        ALPAKA_MAV_ACCESSOR_ASSIGN_OP(*=)


#undef ALPAKA_MAV_ACCESSOR_ASSIGN_OP
    };
} // namespace alpaka::mav::details
