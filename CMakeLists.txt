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
target_include_directories(moore-demo PUBLIC include)

add_executable(utest-test-fixtures src/utest-test-fixtures.cpp)
target_link_libraries(utest-test-fixtures CATCH)
target_include_directories(utest-test-fixtures PUBLIC include)

add_executable(category-Cpp src/category-Cpp.cpp)
target_link_libraries(category-Cpp CATCH)
target_include_directories(category-Cpp PUBLIC include)

add_executable(list-functor src/list-functor.cpp)
target_link_libraries(list-functor CATCH)
target_include_directories(list-functor PUBLIC include)

add_executable(maybe-optional-functor src/maybe-optional-functor.cpp)
target_link_libraries(maybe-optional-functor CATCH)
target_include_directories(maybe-optional-functor PUBLIC include)

# add_executable(filename src/filename.cpp)
# target_link_libraries(filename CATCH)
# target_include_directories(filename PUBLIC include)
