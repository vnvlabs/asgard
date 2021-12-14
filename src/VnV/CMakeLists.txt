set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(AFTER ${CMAKE_BINARY_DIR})
add_library(adaptivity SHARED "")

target_sources(adaptivity PRIVATE adaptivity.cpp)
link_vnv_file(adaptivity MeshWatcher cpp)
