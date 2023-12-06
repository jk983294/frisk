file(GLOB CommonSrc *.cpp path/*.cpp point/*.cpp point/internal/*.cpp)

add_library(frisk_common STATIC ${CommonSrc})
target_link_libraries( frisk_common ${OpenMP_CXX_LIBRARIES} pthread )
set_property(TARGET frisk_common PROPERTY POSITION_INDEPENDENT_CODE ON)
#install(TARGETS teformula_static RUNTIME DESTINATION lib)