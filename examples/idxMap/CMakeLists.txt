set(TARGET_EXAMPLE_IDX_MAP idxMap)
add_executable(${TARGET_EXAMPLE_IDX_MAP})
target_sources(${TARGET_EXAMPLE_IDX_MAP}
        PRIVATE main.cpp)

target_include_directories(${TARGET_EXAMPLE_IDX_MAP} PRIVATE include)
target_link_libraries(${TARGET_EXAMPLE_IDX_MAP} PUBLIC alpaka::mav)
alpaka_finalize(${TARGET_EXAMPLE_IDX_MAP})
