#pragma once

#include <alpaka/alpaka.hpp>

#include <string>
#include <utility>


// print data of a 1D DataSource
std::string dataSourceToStr(alpaka::concepts::IDataSource auto &data) {
    static_assert(ALPAKA_TYPEOF(data)::dim() == 1);
    std::stringstream ss;
    ss << "[";
    for (auto i = 0; i < data.getExtents()[0]; ++i) {
        ss << data[i] << " ";
    }
    ss << "]";
    return ss.str();
}

// return a list of element indices where the next element of a element is bigger or smaller than 1
template<alpaka::concepts::IDataSource TDataSource, typename TIdx = TDataSource::index_type>
std::vector<TIdx> getIndexBreaks(TDataSource& data)
{
    std::vector<TIdx> indices;
    for(TIdx i = 0; i < data.getExtents()[0] - 1; ++i)
    {
        if(std::abs(data[i] - data[i + 1]) > 1)
        {
            indices.push_back(i);
        }
    }

    return indices;
}

// render data to string
// only display the elements which are defined in indices_breaks and 2 elements befor and after
// return also for each postion in indices_breaks the relative difference in number of characters to the it's prevision
// postion
//  for the first element, return the distance the index break and the beginn of the line
template<alpaka::concepts::IDataSource TDataSource, typename TIdx = TDataSource::index_type>
std::pair<std::string, std::vector<TIdx>> renderVector(TDataSource& data, std::vector<TIdx> indices_breaks)
{
    std::vector<TIdx> local_rendered_indices_pos{};
    std::vector<TIdx> global_rendered_indices_pos{};
    std::stringstream ss;
    std::string prefix{"["};

    TIdx render_index_offset = 0;
    for(TIdx i = 0; i < indices_breaks.size(); ++i)
    {
        TIdx index_break = indices_breaks[i];
        std::stringstream local_ss{};

        if(i == 0)
        {
            local_ss << prefix;
            if(index_break > 2)
            {
                local_ss << ".., " << data[index_break - 2] << ", " << data[index_break - 1] << ", ";
            }
            else
            {
                for(TIdx j = 0; j < index_break; ++j)
                {
                    local_ss << data[j] << ", ";
                }
            }
            TIdx cursor_pos = render_index_offset + local_ss.str().size() + 1;
            global_rendered_indices_pos.push_back(cursor_pos);
            local_rendered_indices_pos.push_back(cursor_pos);

            local_ss << data[index_break] << ", ";

            render_index_offset = local_ss.str().size() - cursor_pos;
            ss << local_ss.str();
            continue;
        }

        TIdx index_break_before = indices_breaks[i - 1];
        TIdx index_diff = index_break - index_break_before;
        if(index_diff > 4)
        {
            ss << data[index_break_before + 1] << ", " << data[index_break_before + 2] << ", " << ".., "
               << data[index_break - 2] << ", " << data[index_break - 1] << ", ";
        }
        else
        {
            for(TIdx j = index_diff; j > 0; --j)
            {
                if((index_break - j) > index_break_before)
                {
                    ss << data[index_break - j] << ", ";
                }
            }
        }
        local_rendered_indices_pos.push_back(ss.str().size() - global_rendered_indices_pos.back() + 1);
        global_rendered_indices_pos.push_back(ss.str().size() + 1);
        ss << data[index_break] << ", ";
    }

    TIdx last_index_break = indices_breaks.back();
    TIdx last_index = data.getExtents()[0] - 1;
    TIdx missing_elements = last_index - last_index_break;

    if(missing_elements > 4)
    {
        ss << data[last_index_break + 1] << ", " << data[last_index_break + 2] << ", " << ".., "
           << data[last_index - 1] << ", ";
    }
    else
    {
        for(TIdx j = last_index_break + 1; j < last_index; ++j)
        {
            ss << data[j] << ", ";
        }
    }

    ss << data[last_index] << "]";

    return {ss.str(), local_rendered_indices_pos};
}

// return rendered element postions
std::string renderIndexPos(auto rendered_indices_pos, auto indices_breaks)
{
    std::stringstream ss;

    typename decltype(indices_breaks)::size_type offset = 0;
    for(auto i = 0; i < indices_breaks.size(); ++i)
    {
        std::string index = std::to_string(indices_breaks[i]);
        ss << std::string(rendered_indices_pos[i] - 1 - offset, ' ') << indices_breaks[i];
        offset = index.size() - 1;
    }
    ss << "\n";

    for(auto e : rendered_indices_pos)
    {
        ss << std::string(e - 1, ' ') << "|";
    }
    ss << "\n";
    for(auto e : rendered_indices_pos)
    {
        ss << std::string(e - 1, ' ') << "V";
    }
    return ss.str();
}

// display elements of data source, if the difference between a element and it's successor is bigger than 1
// also display the index postion of this element
std::string dataSourceToStrSparse(alpaka::concepts::IDataSource auto& data)
{
    static_assert(ALPAKA_TYPEOF(data)::dim() == 1);

    auto indices_breaks = getIndexBreaks(data);

    auto [rendered_vector, rendered_indices_pos] = renderVector(data, indices_breaks);

    std::string rendered_index = renderIndexPos(rendered_indices_pos, indices_breaks);

    std::stringstream ss;
    ss << rendered_index << "\n" << rendered_vector;

    return ss.str();
}
