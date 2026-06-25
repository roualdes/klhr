add_executable(example example.cpp)
target_include_directories(example PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(example PRIVATE
                              klhr
                              bfgs
                              mcmcpp
                              dylib
                              Eigen3::Eigen
                              OpenMP::OpenMP_CXX
                              ${KLHR_HIGHFIVE_TARGET}
                              ${HDF5_LIBRARIES}
)
target_compile_options(example PRIVATE -Wall -Wextra)
