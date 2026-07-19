FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.5.0
)
FetchContent_MakeAvailable(cli11)

add_executable(klhr-experiment experiment.cpp)
target_link_libraries(klhr-experiment PRIVATE
                              klhr
                              bfgs
                              mcmcpp
                              dylib
                              Eigen3::Eigen
                              HighFive::HighFive
                              CLI11::CLI11
                              nlohmann_json::nlohmann_json
)
target_compile_options(klhr-experiment PRIVATE -Wall -Wextra)

include(GNUInstallDirs)

install(
    TARGETS klhr-experiment
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
