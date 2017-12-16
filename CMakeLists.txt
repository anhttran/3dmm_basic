cmake_minimum_required (VERSION 2.6)
project (CLM_framework)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:\$ORIGIN/../bin:\$ORIGIN")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
# OpenCV
if(UNIX)
#	set (OpenCV_DIR "/home/anh/libs/opencv-2.4.9/release")
#	find_package( OpenCV REQUIRED HINTS "/usr/include" )
#	MESSAGE("OpenCV information:") 
#	MESSAGE("  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}") 
#	MESSAGE("  OpenCV_LIBRARIES: ${OpenCV_LIBS}") 
	find_package( Boost REQUIRED COMPONENTS filesystem system)
	MESSAGE("Boost information:") 
	#SET(Boost_INCLUDE_DIRS "/usr/include")
	#SET(Boost_LIBRARIES "/usr/lib64/libboost_filesystem.so;/usr/lib64/libboost_system.so")	
	#SET(Boost_LIBRARY_DIRS "/usr/lib64")
	MESSAGE("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}") 
	MESSAGE("  Boost_LIBRARIES: ${Boost_LIBRARIES}") 
	MESSAGE("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}") 

	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS}/boost)
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})	

        find_package( OpenCV REQUIRED )
        #SET(OpenCV_INCLUDE_DIRS "/usr/include/opencv;/usr/include")
        #SET(OpenCV_LIBS "debug;/usr/lib64/libopencv_calib3d.so;debug;/usr/lib64/libopencv_contrib.so;debug;/usr/lib64/libopencv_core.so;debug;/usr/lib64/libopencv_features2d.so;debug;/usr/lib64/libopencv_flann.so;debug;/usr/lib64/libopencv_highgui.so;debug;/usr/lib64/libopencv_imgproc.so;debug;/usr/lib64/libopencv_legacy.so;debug;/usr/lib64/libopencv_ml.so;debug;/usr/lib64/libopencv_objdetect.so;debug;/usr/lib64/libopencv_photo.so;debug;/usr/lib64/libopencv_stitching.so;debug;/usr/lib64/libopencv_superres.so;debug;/usr/lib64/libopencv_ts.so;debug;/usr/lib64/libopencv_video.so;debug;/usr/lib64/libopencv_videostab.so;optimized;/usr/lib64/libopencv_calib3d.so;optimized;/usr/lib64/libopencv_contrib.so;optimized;/usr/lib64/libopencv_core.so;optimized;/usr/lib64/libopencv_features2d.so;optimized;/usr/lib64/libopencv_flann.so;optimized;/usr/lib64/libopencv_highgui.so;optimized;/usr/lib64/libopencv_imgproc.so;optimized;/usr/lib64/libopencv_legacy.so;optimized;/usr/lib64/libopencv_ml.so;optimized;/usr/lib64/libopencv_objdetect.so;optimized;/usr/lib64/libopencv_photo.so;optimized;/usr/lib64/libopencv_stitching.so;optimized;/usr/lib64/libopencv_superres.so;optimized;/usr/lib64/libopencv_ts.so;optimized;/usr/lib64/libopencv_video.so;optimized;/usr/lib64/libopencv_videostab.so")

        MESSAGE("OpenCV information:")
	MESSAGE("  OpenCV_DIR: ${OpenCV_DIR}")
        MESSAGE("  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}") 
        MESSAGE("  OpenCV_LIBRARIES: ${OpenCV_LIBS}")


endif(UNIX)

# Move CLM model
#file(GLOB files "lib/local/CLM/model/*.txt")
#foreach(file ${files})
	if (MSVC)
		file(COPY modules/CLM/model DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
		file(COPY modules/CLM/model DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)
		file(COPY modules/CLM/classifiers DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
		file(COPY modules/CLM/classifiers DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)
	else(MSVC)
		file(COPY modules/CLM/model DESTINATION ${CMAKE_BINARY_DIR}/bin)
		file(COPY modules/CLM/classifiers DESTINATION ${CMAKE_BINARY_DIR}/bin)
	endif(MSVC)
#endforeach()

# Move OpenCV classifiers
#file(GLOB files "lib/3rdParty/OpenCV/classifiers/*.xml")
#foreach(file ${files})
#	if (MSVC)
#		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/classifiers)
#		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/classifiers)
#	else(MSVC)
#		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/classifiers)
#	endif(MSVC)
#endforeach()

#file(GLOB files "lib/BaselFace.dat")
#foreach(file ${files})
	if (MSVC)
		file(COPY lib/BaselFace.dat DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
		file(COPY lib/BaselFace.dat DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)
	else(MSVC)
		file(COPY lib/BaselFace.dat DESTINATION ${CMAKE_BINARY_DIR}/bin)
	endif(MSVC)
#endforeach()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 4.7)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif ()

# Boost
if(WIN32)
	include_directories(lib/3rdParty/boost)
	include_directories(lib/3rdParty/boost/boost)
	link_directories( ${PROJECT_SOURCE_DIR}/lib/3rdParty/boost/lib )
else()
	INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})
endif()

# OpenMP

find_package(OpenMP)
if (OPENMP_FOUND)
	message("OPENMP FOUND")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# Eigen
include_directories(lib/3rdParty/Eigen)

# libraries (ordering matters)
add_subdirectory(modules/CvGl)
#add_subdirectory(exe/EPnP)
add_subdirectory(modules/CLM)

# executables
add_subdirectory(config)
#add_subdirectory(exe/SimpleCLM)
