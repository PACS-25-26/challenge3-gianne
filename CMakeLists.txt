cmake_minimum_required(VERSION 3.16)
project(LaplaceSolver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find MPI ----
find_package(MPI REQUIRED)

# ---- Find OpenMP ----
find_package(OpenMP REQUIRED)

# ---- Source files ----
set(SOURCES
    main.cpp
    Grid2D.cpp
    LaplaceSolver.cpp
)

# ---- Build the executable ----
add_executable(laplace_solver ${SOURCES})

# ---- Link MPI and OpenMP ----
target_link_libraries(laplace_solver
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

# ---- Include MPI headers ----
target_include_directories(laplace_solver PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# ---- Compiler flags: optimisation + OpenMP ----
target_compile_options(laplace_solver PRIVATE
    -O2
    -Wall
)
