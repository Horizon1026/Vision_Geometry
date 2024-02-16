aux_source_directory( . AUX_SRC_VISION_GEOMETRY_ICP )

# Add all components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

# Add kd-tree of slam utility data struct.
if ( NOT TARGET lib_kd_tree )
    add_subdirectory( ${SLAM_UTILITY_PATH}/data_struct/kd_tree ${PROJECT_SOURCE_DIR}/build/lib_kd_tree )
endif()

add_library( lib_vision_geometry_icp ${AUX_SRC_VISION_GEOMETRY_ICP} )
target_include_directories( lib_vision_geometry_icp PUBLIC . )
target_link_libraries( lib_vision_geometry_icp
    lib_slam_utility_datatype
    lib_slam_utility_math
    lib_slam_utility_operate
    lib_slam_utility_log

    lib_kd_tree
)
