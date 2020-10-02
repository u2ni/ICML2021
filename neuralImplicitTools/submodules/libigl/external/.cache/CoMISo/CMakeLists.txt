# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(CoMISo-download NONE)

include(ExternalProject)
ExternalProject_Add(CoMISo-download
                    TLS_VERIFY;OFF;GIT_CONFIG;advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/CoMISo.git;GIT_TAG;1f9618cf9b7bd77370d817976470d59091928606
                    SOURCE_DIR          "/mnt/school/shapeMemory/submodules/libigl/cmake/../external/CoMISo"
                    BINARY_DIR          "/mnt/school/shapeMemory/submodules/libigl/build/CoMISo-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
