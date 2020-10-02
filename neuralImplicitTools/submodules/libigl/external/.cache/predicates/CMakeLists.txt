# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(predicates-download NONE)

include(ExternalProject)
ExternalProject_Add(predicates-download
                    TLS_VERIFY;OFF;GIT_CONFIG;advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/libigl-predicates.git;GIT_TAG;4c57c1d3f31646b010d1d58bfbe201e75c2b2ad8
                    SOURCE_DIR          "/mnt/school/shapeMemory/submodules/libigl/cmake/../external/predicates"
                    BINARY_DIR          "/mnt/school/shapeMemory/submodules/libigl/build/predicates-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
