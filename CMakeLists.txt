cmake_minimum_required(VERSION 3.11...3.31)

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

set(SOURCES
  include/onlinepca.hpp
  include/windowedadaptation.hpp
  include/lbfgstransport.hpp
  include/hdf5drawwriter.hpp
  include/klhr.hpp
  include/gausshermite.hpp
  include/gausslaguerre.hpp
  include/welford.hpp
)

include(FetchContent)

## Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        5.0.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)


## dylib
FetchContent_Declare(
    dylib
    GIT_REPOSITORY "https://github.com/martin-olivier/dylib"
    GIT_TAG        "v3.0.1"
)
FetchContent_MakeAvailable(dylib)

## mcmcpp
add_library(mcmcpp INTERFACE IMPORTED)
target_include_directories(mcmcpp INTERFACE
    "/Users/edward/mcmcpp/include"
)

## bfgs
add_library(bfgs INTERFACE IMPORTED)
target_include_directories(mcmcpp INTERFACE
    "/Users/edward/bfgs/include"
)

## HDF5
# HDF5 need be installed and findable
find_package(HDF5 REQUIRED COMPONENTS C)

set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    highfive
    GIT_REPOSITORY "https://github.com/highfive-devs/highfive"
    GIT_TAG        "main"
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(highfive)
set(KLHR_HIGHFIVE_TARGET HighFive::HighFive)

## Examples
if (KLHR_BUILD_EXAMPLES)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/examples")
endif()

# klhr
add_library(klhr INTERFACE)
target_include_directories(klhr INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # for builds
)
