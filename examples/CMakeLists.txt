FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.5.0
)
FetchContent_MakeAvailable(cli11)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE
                              klhr
                              bfgs
                              mcmcpp
                              dylib
                              Eigen3::Eigen
                              CLI11::CLI11
)
target_compile_options(example PRIVATE -Wall -Wextra)

add_executable(mala mala.cpp)
target_link_libraries(mala PRIVATE
                           klhr
                           bfgs
                           mcmcpp
                           dylib
                           Eigen3::Eigen
                           CLI11::CLI11
)
target_compile_options(mala PRIVATE -Wall -Wextra)

add_executable(barker barker.cpp)
target_link_libraries(barker PRIVATE
                              klhr
                              bfgs
                              mcmcpp
                              dylib
                              Eigen3::Eigen
                              CLI11::CLI11
)
target_compile_options(barker PRIVATE -Wall -Wextra)
