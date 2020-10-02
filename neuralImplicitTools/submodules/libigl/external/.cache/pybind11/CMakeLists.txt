# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(pybind11-download NONE)

include(ExternalProject)
ExternalProject_Add(pybind11-download
                    TLS_VERIFY;OFF;GIT_CONFIG;advice.detachedHead=false;GIT_REPOSITORY;https://github.com/pybind/pybind11.git;GIT_TAG;2d0507db43cd5a117f7843e053b17dffca114107
                    SOURCE_DIR          "/mnt/school/shapeMemory/submodules/libigl/cmake/../external/pybind11"
                    BINARY_DIR          "/mnt/school/shapeMemory/submodules/libigl/build/pybind11-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
