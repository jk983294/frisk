find_package(pybind11 REQUIRED)

pybind11_add_module(pyelnet "PyElNet.cpp")
target_link_libraries( pyelnet PRIVATE ${CMAKE_THREAD_LIBS_INIT} frisk_common z rt dl)
