FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.5.0
)
FetchContent_MakeAvailable(cli11)

add_executable(example example.cpp)
target_include_directories(example PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(example PRIVATE
                              klhr
                              bfgs
                              mcmcpp
                              dylib
                              Eigen3::Eigen
                              ${KLHR_HIGHFIVE_TARGET}
                              ${HDF5_LIBRARIES}
                              CLI11::CLI11
)
target_compile_options(example PRIVATE -Wall -Wextra)
