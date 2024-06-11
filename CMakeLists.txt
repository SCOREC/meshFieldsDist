cmake_minimum_required(VERSION 3.13.0)

project(meshFieldsDist VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Omega_h REQUIRED)
#Clear the omegah compilation flags that it passes to cuda.  Using the
# kokkos target, and nvcc_wrapper, provide sufficient flags.
set_property(TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS "") #try to remove?

find_package(meshFields REQUIRED)

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

add_executable(testDist testDist.cpp)
target_link_libraries(testDist PRIVATE meshFields::meshFields Omega_h::omega_h)

add_executable(syncRankThree syncRankThree.cpp)
target_link_libraries(syncRankThree PRIVATE meshFields::meshFields Omega_h::omega_h)

add_test(testDist_plate_2p mpirun -np 2 ./testDist ${CMAKE_SOURCE_DIR}/testMeshes/tri8_2p.osh tri2p.vtk)
add_test(testDist_plate_4p mpirun -np 4 ./testDist ${CMAKE_SOURCE_DIR}/testMeshes/tri8_4p.osh tri4p.vtk)


