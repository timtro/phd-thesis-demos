cmake_minimum_required(VERSION 3.9.1)
project(mealy-control-demo)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_STANDARD 17)

message("C++ Compiler : ${CMAKE_CXX_COMPILER}")
message("C++ Flags    : ${CMAKE_CXX_FLAGS}")
message("CXX Standard : ${CMAKE_CXX_STANDARD}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(mealy-demo src/mealy-fp.cpp)
