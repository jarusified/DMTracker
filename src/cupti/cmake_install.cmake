# Install script for directory: /home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/libkineto.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kineto" TYPE FILE FILES
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/AbstractConfig.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/ActivityProfilerInterface.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/ActivityTraceInterface.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/ActivityType.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/Config.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/ClientInterface.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/GenericTraceActivity.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/IActivityProfiler.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/ILoggerObserver.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/ITraceActivity.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/TraceSpan.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/ThreadUtil.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/libkineto.h"
    "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/include/time_since_epoch.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/kineto/kinetoLibraryConfig.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/kineto/kinetoLibraryConfig.cmake"
         "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/CMakeFiles/Export/share/cmake/kineto/kinetoLibraryConfig.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/kineto/kinetoLibraryConfig-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/kineto/kinetoLibraryConfig.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/kineto" TYPE FILE FILES "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/CMakeFiles/Export/share/cmake/kineto/kinetoLibraryConfig.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/kineto" TYPE FILE FILES "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/CMakeFiles/Export/share/cmake/kineto/kinetoLibraryConfig-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/fmt/cmake_install.cmake")
  include("/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/test/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/kineto/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
