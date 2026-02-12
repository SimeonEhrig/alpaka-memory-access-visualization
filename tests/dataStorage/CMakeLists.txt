set(TARGET_TEST_MDSPAN test_mav_mdspan)
add_executable(${TARGET_TEST_MDSPAN})
target_sources(${TARGET_TEST_MDSPAN}
        PRIVATE test_mdspan.cpp)

target_link_libraries(${TARGET_TEST_MDSPAN} PRIVATE Catch2::Catch2WithMain alpaka::mav)
alpaka_finalize(${TARGET_TEST_MDSPAN})

add_test(NAME ${TARGET_TEST_MDSPAN} COMMAND ${TARGET_TEST_MDSPAN})

set(TARGET_TEST_ACCESSOR test_mav_accessor)
add_executable(${TARGET_TEST_ACCESSOR})
target_sources(${TARGET_TEST_ACCESSOR}
        PRIVATE test_accessor.cpp)

target_link_libraries(${TARGET_TEST_ACCESSOR} PRIVATE Catch2::Catch2WithMain alpaka::mav)
alpaka_finalize(${TARGET_TEST_ACCESSOR})

add_test(NAME ${TARGET_TEST_ACCESSOR} COMMAND ${TARGET_TEST_ACCESSOR})