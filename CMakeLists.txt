cmake_minimum_required( VERSION 3.10 )

project( vision_geometry )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

# Add slam utility
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_slam_utility )
endif()

# Add vision geometry
if ( NOT TARGET lib_vision_geometry )
    add_subdirectory( src ${PROJECT_SOURCE_DIR}/build/lib_vision_geometry )
endif()

# Add test_pnp
add_executable( test_pnp
    test/test_pnp.cpp
)
target_link_libraries( test_pnp
    lib_slam_utility
    lib_vision_geometry
)

# Add test_triangulation
add_executable( test_triangulation
    test/test_triangulation.cpp
)
target_link_libraries( test_triangulation
    lib_slam_utility
    lib_vision_geometry
)


# Add test_epipolar
add_executable( test_epipolar
    test/test_epipolar.cpp
)
target_link_libraries( test_epipolar
    lib_slam_utility
    lib_vision_geometry
)
