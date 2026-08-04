add_executable(k-adaptation-test k_adaptation_test.cpp)
target_link_libraries(k-adaptation-test PRIVATE klhr mcmcpp)
target_compile_options(k-adaptation-test PRIVATE -Wall -Wextra)

add_test(NAME k-adaptation COMMAND k-adaptation-test)

add_executable(overrelaxation-test overrelaxation_test.cpp)
target_link_libraries(overrelaxation-test PRIVATE klhr Eigen3::Eigen)
target_compile_options(overrelaxation-test PRIVATE -Wall -Wextra)

add_test(NAME overrelaxation COMMAND overrelaxation-test)

add_executable(quadrature-test quadrature_test.cpp)
target_link_libraries(quadrature-test PRIVATE klhr Eigen3::Eigen)
target_compile_options(quadrature-test PRIVATE -Wall -Wextra)

add_test(NAME quadrature COMMAND quadrature-test)
