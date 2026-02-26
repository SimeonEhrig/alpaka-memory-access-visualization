#pragma once

#include "access_info.hpp"
#include "accessor.hpp"

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
            using AccessMdSpan = alpaka::
                MdSpan<alpaka::mav::AccessInfo<T_Extents>, T_AccessExtents, T_AccessPitches, T_AccessMemAlignment>;

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
            concepts::Vector T_CounterExtents,
            concepts::Vector T_CounterPitches,
            concepts::Alignment T_MemAlignment = Alignment<>,
            concepts::Alignment T_AccessMemAlignment = Alignment<>,
            concepts::Alignment T_CounterMemAlignment = Alignment<>>
        struct MdSpan : public alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>
        {
        private:
            using DataMdSpan = alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>;
            using AccessMdSpan = alpaka::
                MdSpan<alpaka::mav::AccessInfo<T_Extents>, T_AccessExtents, T_AccessPitches, T_AccessMemAlignment>;
            using CounterMdSpan
                = alpaka::MdSpan<typename T_Extents::type, T_CounterExtents, T_CounterPitches, T_CounterMemAlignment>;


            T_Acc const& m_acc;
            T_Extents const m_local_thread_idx;
            T_Extents const m_block_idx;
            AccessMdSpan m_access_mdspan;
            CounterMdSpan m_counter_mdspan;


        public:
            constexpr MdSpan(
                T_Acc const& acc,
                DataMdSpan const& data_mdspan,
                AccessMdSpan const& access_mdspan,
                CounterMdSpan const& counter_md_span)
                : alpaka::MdSpan<T_Type, T_Extents, T_Pitches, T_MemAlignment>(data_mdspan)
                , m_acc(acc)
                , m_local_thread_idx(acc[alpaka::layer::thread].idx())
                , m_block_idx(acc[alpaka::layer::block].idx())
                , m_access_mdspan(access_mdspan)
                , m_counter_mdspan(counter_md_span)
            {
            }

            constexpr auto operator[](concepts::Vector auto const& idx)
            {
                std::size_t const max_counter = m_access_mdspan.getExtents()[0] / this->getExtents()[0];

                return alpaka::mav::details::AccessProxy{
                    &DataMdSpan::operator[](idx),
                    m_local_thread_idx,
                    m_block_idx,
                    m_counter_mdspan[idx],
                    &(m_access_mdspan[alpaka::Vec{idx[0] * max_counter}]),
                    max_counter};
            }
        };
    } // namespace onAcc

} // namespace alpaka::mav
