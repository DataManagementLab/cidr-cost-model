cmake_minimum_required(VERSION 3.1)

# set some default values
if(NOT CMAKE_CXX_STANDARD)
        set (CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -pedantic -pipe -pthread -fopenmp -march=native -g" CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()
if(NOT CMAKE_CXX_FLAGS_DEBUG)
        set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG" CACHE STRING "Flags used by the compiler during Release builds." FORCE)
endif()
if(NOT CMAKE_CXX_FLAGS_RELEASE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during Release builds." FORCE)
endif()
if(NOT CMAKE_EXE_LINKER_FLAGS_RELEASE)
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Wl,-O3" CACHE STRING "Flags used by the linker during release builds." FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

PROJECT(GREYBOX CXX)

add_subdirectory(external)

# Add sources when present and change remove "INTERFACE"
add_library(greybox INTERFACE)

# When library has sources, replace "INTERFACE" with "PUBLIC"
target_include_directories(greybox INTERFACE execution_engine operators storage)

target_link_libraries(greybox INTERFACE nlohmann_json::nlohmann_json)

add_executable(execution_engine_test execution_engine/test.cpp)
add_executable(storage_test storage/testStorage.cpp)
add_executable(operator_test operators/ops_test.cpp)
add_executable(pipeline_cost_exp1 training_data_generator/pipeline_cost_exp1.cpp)
target_link_libraries(storage_test greybox)
target_link_libraries(operator_test greybox)
target_link_libraries(pipeline_cost_exp1 greybox)
target_link_libraries(execution_engine_test greybox)
