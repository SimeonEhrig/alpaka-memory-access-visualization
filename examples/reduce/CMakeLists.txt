set(_REDUCE reduce)
add_executable(${_REDUCE})
target_sources(${_REDUCE}
        PRIVATE reduce.cpp)

target_link_libraries(${_REDUCE} PUBLIC alpaka::mav)
alpaka_finalize(${_REDUCE})