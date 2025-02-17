add_library(gpr
    libs/gpr/gpr.cpp
    libs/gpr/gpr_optimizer.cpp
    libs/gpr/covariance.cpp
    libs/gpr/parameters.cpp
)

target_link_libraries(gpr
    pestpp_common
    Eigen3::Eigen
)


add_executable(pestpp-gpr
    programs/pestpp-gpr.cpp
)

target_link_libraries(pestpp-gpr
    gpr
    pestpp_common
    yamr
    ${EXTRA_LIBS}
)