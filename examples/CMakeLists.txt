add_executable(example example.cpp)
target_link_libraries(example PRIVATE klhr bfgs mcmcpp dylib Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(example PRIVATE -Wall -Wextra)
