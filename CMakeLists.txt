cmake_minimum_required(VERSION 3.12)

if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

project(CppPlayground VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE
        "Debug"
        CACHE
            STRING "Default build type: Debug"
        FORCE
    )
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_definitions( -DDEBUG )
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    add_definitions( -DRELEASE )
endif()

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${CMAKE_SYSTEM_PROCESSOR}/bin
)

set(
    CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${CMAKE_SYSTEM_PROCESSOR}/lib
)

set(
    CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${CMAKE_SYSTEM_PROCESSOR}/lib
)

add_subdirectory (src)
