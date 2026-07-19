cmake_minimum_required(VERSION 3.14...3.31)

project(klhr
  VERSION 0.1.0
  DESCRIPTION "KLHR: KL Hit and Run"
  LANGUAGES CXX
)

option(KLHR_BUILD_EXAMPLES "Build the examples target for the library" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

include(FetchContent)

## Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        5.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

## dylib
FetchContent_Declare(
    dylib
    GIT_REPOSITORY https://github.com/martin-olivier/dylib.git
    GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(dylib)

## mcmcpp
FetchContent_Declare(
    mcmcpp
    GIT_REPOSITORY https://github.com/roualdes/mcmcpp.git
    GIT_TAG        a53be58fdfed0109bee684f33ac9b10d1f0e8b45
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(mcmcpp)

## bfgs
FetchContent_Declare(
    bfgs
    GIT_REPOSITORY https://github.com/roualdes/bfgs.git
    GIT_TAG        658035f2e01ac3c74f8094ab3d8c1c357cf287e1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(bfgs)

## HDF5
set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    highfive
    GIT_REPOSITORY "https://github.com/highfive-devs/highfive"
    GIT_TAG        v3.3.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(highfive)

## JSON


FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    )
FetchContent_MakeAvailable(json)

## Examples
if (KLHR_BUILD_EXAMPLES)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/examples")
endif()

## Experiments
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/experiments")

# klhr
add_library(klhr INTERFACE)
target_include_directories(klhr INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # for builds
  $<INSTALL_INTERFACE:include>
)
