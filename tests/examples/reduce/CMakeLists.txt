set(_TEST_REDUCE test_reduce)
add_executable(${_TEST_REDUCE})
target_sources(${_TEST_REDUCE}
        PRIVATE test_reduce.cpp)

target_link_libraries(${_TEST_REDUCE} PRIVATE Catch2::Catch2WithMain alpaka::mav)
target_include_directories(${_TEST_REDUCE} PRIVATE ${CMAKE_SOURCE_DIR}/examples/reduce)
alpaka_finalize(${_TEST_REDUCE})