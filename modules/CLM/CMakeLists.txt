include_directories(${BOOST_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

SET(SOURCE
    src/CCNF_patch_expert.cpp
	src/CLM.cpp
	src/CLM_utils.cpp
    src/Patch_experts.cpp
	src/PAW.cpp
    src/PDM.cpp
	src/CLMTracker.cpp
	src/DetectionValidator.cpp
	src/SVR_patch_expert.cpp
)

SET(HEADERS
    include/CCNF_patch_expert.h
	include/CLM.h
    include/CLMParameters.h
	include/CLM_utils.h
    include/Patch_experts.h
	include/PAW.h	
    include/PDM.h
	include/CLMTracker.h	
	include/SVR_patch_expert.h
	include/DetectionValidator.h	
)

include_directories(./include)
include_directories(${CLM_SOURCE_DIR}/include)

add_library( CLM SHARED ${SOURCE} ${HEADERS})
target_link_libraries(CLM ${OpenCV_LIBS})

install (TARGETS CLM DESTINATION .)
install (FILES ${HEADERS} DESTINATION include)
install (DIRECTORY classifiers DESTINATION .)
install (DIRECTORY model DESTINATION .)
install (FILES ../../lib/BaselFace.dat DESTINATION .)
