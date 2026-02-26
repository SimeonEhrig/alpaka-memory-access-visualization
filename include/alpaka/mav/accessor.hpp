#pragma once

namespace alpaka::mav::details
{
    template<typename TData, typename TExtents, typename TAccessInfo>
    class AccessProxy
    {
        TData* const m_ptr;
        TExtents& m_local_thread_index;
        TExtents& m_block_index;
        std::size_t& m_access_counter;
        TAccessInfo* const m_access_infos;
        std::size_t m_max_access_number;

    public:
        constexpr AccessProxy(
            TData* const ptr,
            TExtents& local_thread_index,
            TExtents& block_index,
            std::size_t& access_counter,
            TAccessInfo* const access_infos,
            std::size_t const max_access_number)
            : m_ptr(ptr)
            , m_local_thread_index(local_thread_index)
            , m_block_index(block_index)
            , m_access_counter(access_counter)
            , m_access_infos(access_infos)
            , m_max_access_number(max_access_number)
        {
        }

        // write access
        constexpr AccessProxy& operator=(TData const& value)
        {
            *m_ptr = value;
            if(m_access_counter < m_max_access_number)
            {
                m_access_infos[m_access_counter].local_thread_index = m_local_thread_index;
                m_access_infos[m_access_counter].block_index = m_block_index;
                m_access_infos[m_access_counter].read_write = false;
            }
            m_access_counter += 1;
            return *this;
        }

        // read access
        constexpr operator TData() const
        {
            if(m_access_counter < m_max_access_number)
            {
                m_access_infos[m_access_counter].local_thread_index = m_local_thread_index;
                m_access_infos[m_access_counter].block_index = m_block_index;
                m_access_infos[m_access_counter].read_write = true;
            }
            m_access_counter += 1;
            return *m_ptr;
        }

// It's a read/write. Therefore add 2 accesses.
#define ALPAKA_MAV_ACCESSOR_ASSIGN_OP(op)                                                                             \
    constexpr AccessProxy& operator op(TData const& value)                                                            \
    {                                                                                                                 \
        *m_ptr op value;                                                                                              \
        if(m_access_counter < m_max_access_number)                                                                    \
        {                                                                                                             \
            m_access_infos[m_access_counter].local_thread_index = m_local_thread_index;                               \
            m_access_infos[m_access_counter].block_index = m_block_index;                                             \
            m_access_infos[m_access_counter].read_write = true;                                                       \
        }                                                                                                             \
        m_access_counter += 1;                                                                                        \
        if(m_access_counter < m_max_access_number)                                                                    \
        {                                                                                                             \
            m_access_infos[m_access_counter].local_thread_index = m_local_thread_index;                               \
            m_access_infos[m_access_counter].block_index = m_block_index;                                             \
            m_access_infos[m_access_counter].read_write = false;                                                      \
        }                                                                                                             \
        m_access_counter += 1;                                                                                        \
        return *this;                                                                                                 \
    }

        ALPAKA_MAV_ACCESSOR_ASSIGN_OP(+=)
        ALPAKA_MAV_ACCESSOR_ASSIGN_OP(-=)
        ALPAKA_MAV_ACCESSOR_ASSIGN_OP(/=)
        ALPAKA_MAV_ACCESSOR_ASSIGN_OP(*=)


#undef ALPAKA_MAV_ACCESSOR_ASSIGN_OP
    };
} // namespace alpaka::mav::details
