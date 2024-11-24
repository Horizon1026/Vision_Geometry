cmake_minimum_required( VERSION 3.10 )

project( vision_geometry )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

# Add all relative components of vision geometry.
if ( NOT TARGET lib_vision_geometry_epipolar )
    add_subdirectory( src/epipolar ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_epipolar )
endif()
if ( NOT TARGET lib_vision_geometry_pnp )
    add_subdirectory( src/perspective_n_point ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_pnp )
endif()
if ( NOT TARGET lib_vision_geometry_point_triangulator )
    add_subdirectory( src/point_triangulator ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_point_triangulator )
endif()
if ( NOT TARGET lib_vision_geometry_line_triangulator )
    add_subdirectory( src/line_triangulator ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_line_triangulator )
endif()
if ( NOT TARGET lib_vision_geometry_relative_rotation )
    add_subdirectory( src/relative_rotation ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_relative_rotation )
endif()
if ( NOT TARGET lib_vision_geometry_icp )
    add_subdirectory( src/iterative_closest_point ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry_icp )
endif()

# Add visualizor.
set( VISUALIZOR_2D_PATH ${PROJECT_SOURCE_DIR}/../Visualizor2D )
if ( NOT TARGET lib_2d_visualizor )
    add_subdirectory( ${VISUALIZOR_2D_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_2d_visualizor )
endif()
set( VISUALIZOR_3D_PATH ${PROJECT_SOURCE_DIR}/../Visualizor3D )
if ( NOT TARGET lib_3d_visualizor )
    add_subdirectory( ${VISUALIZOR_3D_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_3d_visualizor )
endif()

# Add test_pnp.
add_executable( test_pnp
    test/test_pnp.cpp
)
target_link_libraries( test_pnp
    lib_vision_geometry_pnp
)

# Add test_point_triangulator.
add_executable( test_point_triangulator
    test/test_point_triangulator.cpp
)
target_link_libraries( test_point_triangulator
    lib_vision_geometry_point_triangulator
)

# Add test_line_triangulator.
add_executable( test_line_triangulator
    test/test_line_triangulator.cpp
)
target_link_libraries( test_line_triangulator
    lib_vision_geometry_line_triangulator
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

# Add test_icp.
add_executable( test_icp
    test/test_icp.cpp
)
target_link_libraries( test_icp
    lib_vision_geometry_icp
    lib_3d_visualizor
)
