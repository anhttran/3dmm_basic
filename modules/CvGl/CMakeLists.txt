cmake_minimum_required(VERSION 2.8)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/include")

SET(SOURCE
    FBRender.cpp
)

SET(HEADERS
    cvgl.h
    FBRender.h	
    Material.h
    SimpleMesh.h
    Texture.h	
)

add_library( CvGl SHARED ${SOURCE} ${HEADERS})
target_link_libraries(CvGl -lm -lGLU -lOSMesa)

install (TARGETS CvGl DESTINATION .)
