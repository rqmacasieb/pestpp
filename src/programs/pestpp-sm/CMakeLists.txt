# This CMake file is part of PEST++

add_executable(pestpp-sm pestpp-sm.cpp)

target_compile_options(pestpp-sm PRIVATE ${PESTPP_CXX_WARN_FLAGS})

target_link_libraries(pestpp-sm
  rm_serial
  rm_yamr
  rm_external
)

install(TARGETS pestpp-sm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
