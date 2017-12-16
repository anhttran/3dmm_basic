# Local libraries
include_directories(../modules/CvGl)
include_directories(../modules/CLM/include)
include_directories(../include)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Release)
SET(SOURCE
    ../src/utility.cpp
    ../src/epnp.cpp
    ../src/BaselFace.cpp
    ../src/BaselFaceEstimator.cpp
    ../src/FTModel.cpp
    ../src/FImRenderer.cpp
    ../src/RenderModel.cpp
    ../src/FaceServices2.cpp
#    ../src/testRender.cpp
    ../src/SimpleCLMImg_wexpr.cpp
)

SET(HEADERS
    ../include/utility.h	
    ../include/epnp.h
    ../include/BaselFace.h
    ../include/BaselFaceEstimator.h
    ../include/FTModel.h
    ../include/FImRenderer.h
    ../include/RenderModel.h
    ../include/FaceServices2.h
)
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()
SET(CMAKE_CXX_FLAGS "-w -fopenmp")
add_executable(IRIS_3DMM ${SOURCE} ${HEADERS})

if(WIN32)
	target_link_libraries(IRIS_3DMM ${OpenCVLibraries})
endif(WIN32)
if(UNIX)
	target_link_libraries(IRIS_3DMM ${OpenCV_LIBS} ${Boost_LIBRARIES})
endif(UNIX)
target_link_libraries(IRIS_3DMM -lGLU -lm -lOSMesa)
target_link_libraries(IRIS_3DMM ${CMAKE_BINARY_DIR}/modules/CLM/libCLM.so)
target_link_libraries(IRIS_3DMM ${CMAKE_BINARY_DIR}/modules/CvGl/libCvGl.so)
target_link_libraries(IRIS_3DMM -fopenmp)
#target_link_libraries(IRIS_3DMM /home/anh/Code/3DFaceVideo/Release/exe/EPnP/libEPnP.so)

install (TARGETS IRIS_3DMM DESTINATION .)
