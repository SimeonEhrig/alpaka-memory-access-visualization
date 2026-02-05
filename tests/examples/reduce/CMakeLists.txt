set(TARGET_TEST_REDUCE test_reduce)
add_executable(${TARGET_TEST_REDUCE})
target_sources(${TARGET_TEST_REDUCE}
        PRIVATE test_reduce.cpp)

target_link_libraries(${TARGET_TEST_REDUCE} PRIVATE Catch2::Catch2WithMain alpaka::mav)
target_include_directories(${TARGET_TEST_REDUCE} PRIVATE ${CMAKE_SOURCE_DIR}/examples/reduce)
alpaka_finalize(${TARGET_TEST_REDUCE})

add_test(NAME ${TARGET_TEST_REDUCE} COMMAND ${TARGET_TEST_REDUCE})