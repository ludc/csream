# Install script for directory: /home/denoyer/torch_workspace/csream

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/home/denoyer/torch/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/csream/scm-1/lua/csream" TYPE FILE FILES
    "/home/denoyer/torch_workspace/csream/init.lua"
    "/home/denoyer/torch_workspace/csream/tools/ExperimentLogConsole.lua"
    "/home/denoyer/torch_workspace/csream/tools/ModelsUtils.lua"
    "/home/denoyer/torch_workspace/csream/tools/ExperimentLogCSV.lua"
    "/home/denoyer/torch_workspace/csream/tools/ExperimentLog.lua"
    "/home/denoyer/torch_workspace/csream/modules/MyL1Penalty.lua"
    "/home/denoyer/torch_workspace/csream/modules/MyConstant.lua"
    "/home/denoyer/torch_workspace/csream/modules/GRU.lua"
    "/home/denoyer/torch_workspace/csream/modules/RNN.lua"
    "/home/denoyer/torch_workspace/csream/models/DREAM.lua"
    "/home/denoyer/torch_workspace/csream/models/BREAM.lua"
    "/home/denoyer/torch_workspace/csream/binaries/run_bream.lua"
    "/home/denoyer/torch_workspace/csream/binaries/run_dream.lua"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/denoyer/torch_workspace/csream/build/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/denoyer/torch_workspace/csream/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
