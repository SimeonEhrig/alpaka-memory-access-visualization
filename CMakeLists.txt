cmake_minimum_required(VERSION 3.25)

project(alpaka-memory-access-visualization VERSION 1.0.0)

option(alpaka_MAV_TESTING "Enable/Disable testing" OFF)

find_package(alpaka CONFIG QUIET)
if (NOT alpaka_FOUND)
    include(FetchContent)
    FetchContent_Declare(
            alpaka
            GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git
            GIT_TAG f58f3361e56dada4882bd79540651bb49961e9f7
    )
    FetchContent_MakeAvailable(alpaka)
endif ()

add_library(alpaka_mav INTERFACE)
target_include_directories(alpaka_mav INTERFACE include)
target_link_libraries(alpaka_mav INTERFACE alpaka::alpaka)
add_library(alpaka::mav ALIAS alpaka_mav)

add_subdirectory(examples)

if (alpaka_MAV_TESTING)
    add_subdirectory(tests)
endif ()