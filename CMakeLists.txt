cmake_minimum_required(VERSION 3.25)

project(alpaka-memory-access-visualization VERSION 1.0.0)

option(alpaka_MAV_TESTING "Enable/Disable testing" OFF)
option(alpaka_MAV_EXAMPLES "Enable/Disable examples" OFF)

find_package(alpaka CONFIG QUIET)
if (NOT alpaka_FOUND)
    include(FetchContent)
    FetchContent_Declare(
            alpaka
            GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git
            GIT_TAG 3d491efd92bff57e8bfb5abc51f0316e47c3ae72
    )
    FetchContent_MakeAvailable(alpaka)
endif ()

add_library(alpaka_mav INTERFACE)
target_include_directories(alpaka_mav INTERFACE include)
target_link_libraries(alpaka_mav INTERFACE alpaka::alpaka)
add_library(alpaka::mav ALIAS alpaka_mav)

if (alpaka_MAV_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (alpaka_MAV_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif ()