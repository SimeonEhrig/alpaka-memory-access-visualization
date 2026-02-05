set(TARGET_EXAMPLE_REDUCE reduce)
add_executable(${TARGET_EXAMPLE_REDUCE})
target_sources(${TARGET_EXAMPLE_REDUCE}
        PRIVATE reduce.cpp)

target_include_directories(${TARGET_EXAMPLE_REDUCE} PRIVATE .)
target_link_libraries(${TARGET_EXAMPLE_REDUCE} PUBLIC alpaka::mav)
alpaka_finalize(${TARGET_EXAMPLE_REDUCE})
