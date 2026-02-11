#pragma once

#include "access_info.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka::mav
{
    namespace onHost
    {
        template<
            typename T_Type,
            concepts::Vector T_Extents,
            concepts::Vector T_Pitches,
            concepts::Vector T_AccessExtents,
            concepts::Vector T_AccessPitches,
            concepts::Alignment T_MemAlignment = Alignment<>,
            concepts::Alignment T_AccessMemAlignment = Alignment<>>
        struct MdSpan : public alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>
        {
        private:
            using DataMdSpan = alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>;
            using AccessMdSpan = alpaka::MdSpan<
                alpaka::mav::AccessInfo<T_Extents, T_Extents>,
                T_AccessExtents,
                T_AccessPitches,
                T_AccessMemAlignment>;

            AccessMdSpan m_access_mdspan;
            std::string m_name;

        public:
            constexpr MdSpan(std::string_view name, DataMdSpan const& data_mdspan, AccessMdSpan const& access_mdspan)
                : alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>(data_mdspan)
                , m_name(name)
                , m_access_mdspan(access_mdspan)
            {
            }

            AccessMdSpan& get_access_mdspan()
            {
                return m_access_mdspan;
            }

            AccessMdSpan const& get_access_mdspan() const
            {
                return m_access_mdspan;
            }

            constexpr DataMdSpan::pointer ptr(concepts::Vector auto const& idx)
            {
                return DataMdSpan::ptr(idx);
            }

            constexpr DataMdSpan::const_pointer ptr(concepts::Vector auto const& idx) const
            {
                return DataMdSpan::ptr(idx);
            }

            constexpr std::string_view get_name() const
            {
                return m_name;
            }
        };

        namespace concepts
        {
            // TODO: instead checking for a concrete implementation, the concept should describe the interface
            template<typename T>
            concept MdSpan = alpaka::isSpecializationOf_v<T, alpaka::mav::onHost::MdSpan>;
        } // namespace concepts
    } // namespace onHost

    namespace onAcc
    {
        template<
            typename T_Type,
            concepts::Vector T_Extents,
            concepts::Vector T_Pitches,
            alpaka::onAcc::concepts::Acc T_Acc,
            concepts::Vector T_AccessExtents,
            concepts::Vector T_AccessPitches,
            concepts::Alignment T_MemAlignment = Alignment<>,
            concepts::Alignment T_AccessMemAlignment = Alignment<>>
        struct MdSpan : public alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>
        {
        private:
            using DataMdSpan = alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>;
            using AccessMdSpan = alpaka::MdSpan<
                alpaka::mav::AccessInfo<T_Extents, T_Extents>,
                T_AccessExtents,
                T_AccessPitches,
                T_AccessMemAlignment>;

            T_Acc const& m_acc;
            T_Extents const m_local_thread_idx;
            T_Extents const m_block_idx;
            AccessMdSpan m_access_mdspan;

        public:
            constexpr MdSpan(T_Acc const& acc, DataMdSpan const& data_mdspan, AccessMdSpan const& access_mdspan)
                : alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>(data_mdspan)
                , m_acc(acc)
                , m_local_thread_idx(acc[alpaka::layer::thread].idx())
                , m_block_idx(acc[alpaka::layer::block].idx())
                , m_access_mdspan(access_mdspan)
            {
            }

            constexpr DataMdSpan::reference operator[](concepts::Vector auto const& idx)
            {
                m_access_mdspan[idx].data_index = idx;
                m_access_mdspan[idx].local_thread_index = m_local_thread_idx;
                m_access_mdspan[idx].block_index = m_block_idx;
                m_access_mdspan[idx].access_counter += 1;
                return DataMdSpan::operator[](idx);
            }
        };
    } // namespace onAcc

} // namespace alpaka::mav
