file(GLOB AllSrc "*.cpp")
add_library(elnet SHARED ${AllSrc})
target_link_libraries(elnet frisk_common dl pthread)
set_property(TARGET elnet PROPERTY POSITION_INDEPENDENT_CODE ON)
install(TARGETS elnet RUNTIME DESTINATION lib)
