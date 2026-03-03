option(alpaka_MAV_EXAMPLE_IDXMAP_MATPLOT "Render data in graph.png. Requires gnuplot." OFF)

set(TARGET_EXAMPLE_IDX_MAP idxMap)
add_executable(${TARGET_EXAMPLE_IDX_MAP})
target_sources(${TARGET_EXAMPLE_IDX_MAP}
        PRIVATE main.cpp)

target_include_directories(${TARGET_EXAMPLE_IDX_MAP} PRIVATE include)
target_link_libraries(${TARGET_EXAMPLE_IDX_MAP} PUBLIC alpaka::mav)
alpaka_finalize(${TARGET_EXAMPLE_IDX_MAP})

if (alpaka_MAV_EXAMPLE_IDXMAP_MATPLOT)
    message(STATUS "enable matplotlib++")
    FetchContent_Declare(matplotplusplus
            GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
            GIT_TAG origin/master)
    FetchContent_GetProperties(matplotplusplus)
    if (NOT matplotplusplus_POPULATED)
        FetchContent_Populate(matplotplusplus)
        add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()

    target_link_libraries(${TARGET_EXAMPLE_IDX_MAP} PRIVATE matplot)
    target_compile_options(${TARGET_EXAMPLE_IDX_MAP} PRIVATE "-DDRAW_GRAPH_PNG")
endif ()
