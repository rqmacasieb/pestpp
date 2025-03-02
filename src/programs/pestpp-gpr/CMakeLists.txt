# This CMake file is part of PEST++

add_executable(pestpp-gpr pestpp-gpr.cpp)

target_compile_options(pestpp-gpr PRIVATE ${PESTPP_CXX_WARN_FLAGS})

target_link_libraries(pestpp-gpr
  rm_serial
  rm_yamr
  pestpp_common
)

install(TARGETS pestpp-gpr RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})