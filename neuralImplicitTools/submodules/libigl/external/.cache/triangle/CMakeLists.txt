# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(triangle-download NONE)

include(ExternalProject)
ExternalProject_Add(triangle-download
                    TLS_VERIFY;OFF;GIT_CONFIG;advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/triangle.git;GIT_TAG;d284c4a843efac043c310f5fa640b17cf7d96170
                    SOURCE_DIR          "/mnt/school/shapeMemory/submodules/libigl/cmake/../external/triangle"
                    BINARY_DIR          "/mnt/school/shapeMemory/submodules/libigl/build/triangle-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
