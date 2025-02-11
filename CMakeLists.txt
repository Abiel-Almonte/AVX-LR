cmake_minimum_required(VERSION 3.12.)

project(logisitcRegression VERSION 1.0.0 DESCRIPTION "A low latency implementation of Logistic Regession" LANGUAGES CXX)
set(PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Proejct Description: ${PROJECT_DESCRIPTION}")
message(STATUS "Project SRC Directory: ${PROJECT_SOURCE_DIR}\n")

file(GLOB_RECURSE source_files "${PROJECT_SOURCE_DIR}/*.c??")

message(STATUS "Source files: ${source_files}")

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)

add_executable(run "${source_files}")
target_compile_options(run PRIVATE "-Wall" "-march=native")