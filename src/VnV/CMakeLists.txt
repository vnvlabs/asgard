set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(adaptivity SHARED "")

target_sources(adaptivity PRIVATE adaptivity.cpp)
link_vnv_file(adaptivity MeshWatcher cpp)

#find_package(MPI REQUIRED)

#target_include_directories(adaptivity PRIVATE ${MPI_CXX_INCLUDE_PATH})
#target_compile_options(adaptivity PRIVATE ${MPI_CXX_COMPILE_FLAGS})
#target_link_libraries(adaptivity PRIVATE ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
