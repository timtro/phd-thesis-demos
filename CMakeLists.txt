cmake_minimum_required(VERSION 3.9.1)
project(mealy-control-demo)

set(CMAKE_CXX_FLAGS "-Wall -pedantic")
set(CMAKE_CXX_STANDARD 17)

message("C++ Compiler : ${CMAKE_CXX_COMPILER}")
message("C++ Flags    : ${CMAKE_CXX_FLAGS}")
message("CXX Standard : ${CMAKE_CXX_STANDARD}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(Boost 1.58 COMPONENTS variant REQUIRED)

add_library (CATCH src/catch.cpp)

add_executable(moore-demo src/moore-fp.cpp)
target_link_libraries(moore-demo CATCH)
