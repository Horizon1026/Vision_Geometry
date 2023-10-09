cmake_minimum_required( VERSION 3.10 )

project( vision_geometry )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

# Add all components of vision geometry.
if ( NOT TARGET lib_vision_geometry_epipolar )
    add_subdirectory( src/epipolar ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_epipolar )
endif()
if ( NOT TARGET lib_vision_geometry_pnp )
    add_subdirectory( src/perspective_n_point ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_pnp )
endif()
if ( NOT TARGET lib_vision_geometry_triangulation )
    add_subdirectory( src/triangulation ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_triangulation )
endif()
if ( NOT TARGET lib_vision_geometry_relative_rotation )
    add_subdirectory( src/relative_rotation ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_relative_rotation )
endif()

# Add test_pnp.
add_executable( test_pnp
    test/test_pnp.cpp
)
target_link_libraries( test_pnp
    lib_vision_geometry_pnp
)

# Add test_triangulation.
add_executable( test_triangulation
    test/test_triangulation.cpp
)
target_link_libraries( test_triangulation
    lib_vision_geometry_triangulation
)

# Add test_epipolar.
add_executable( test_epipolar
    test/test_epipolar.cpp
)
target_link_libraries( test_epipolar
    lib_vision_geometry_epipolar
)

# Add test_relative_rotation.
add_executable( test_relative_rotation
    test/test_relative_rotation.cpp
)
target_link_libraries( test_relative_rotation
    lib_vision_geometry_relative_rotation
)
