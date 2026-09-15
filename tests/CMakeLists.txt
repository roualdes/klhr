add_executable(overrelaxation-test overrelaxation_test.cpp)
target_link_libraries(overrelaxation-test PRIVATE klhr Eigen3::Eigen)
target_compile_options(overrelaxation-test PRIVATE -Wall -Wextra)

add_test(NAME overrelaxation COMMAND overrelaxation-test)

add_executable(quadrature-test quadrature_test.cpp)
target_link_libraries(quadrature-test PRIVATE klhr Eigen3::Eigen)
target_compile_options(quadrature-test PRIVATE -Wall -Wextra)

add_test(NAME quadrature COMMAND quadrature-test)
