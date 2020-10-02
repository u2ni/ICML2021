# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(eigen-download NONE)

include(ExternalProject)
ExternalProject_Add(eigen-download
                    TLS_VERIFY;OFF;GIT_CONFIG;advice.detachedHead=false;GIT_REPOSITORY;https://github.com/eigenteam/eigen-git-mirror.git;GIT_TAG;3.2.10
                    SOURCE_DIR          "/mnt/school/shapeMemory/submodules/libigl/cmake/../external/eigen"
                    BINARY_DIR          "/mnt/school/shapeMemory/submodules/libigl/build/eigen-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
