set(TARGET_EXAMPLE_EVEN_ODD even_odd)
add_executable(${TARGET_EXAMPLE_EVEN_ODD})
target_sources(${TARGET_EXAMPLE_EVEN_ODD}
        PRIVATE even_odd.cpp)

target_include_directories(${TARGET_EXAMPLE_EVEN_ODD} PRIVATE .)
target_link_libraries(${TARGET_EXAMPLE_EVEN_ODD} PUBLIC alpaka::mav)
alpaka_finalize(${TARGET_EXAMPLE_EVEN_ODD})
