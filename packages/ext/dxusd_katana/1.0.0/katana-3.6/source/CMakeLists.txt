cmake_minimum_required (VERSION 3.2)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../Resources" CACHE PATH "Out Path")

project(DexterKatanaOps)

#-------------------------------------------------------------------------------
# change KATANA_ROOT for each versions
set(KATANA_ROOT "/opt/Katana3.6v2" CACHE PATH "Path to Katana")

#-------------------------------------------------------------------------------
list(INSERT CMAKE_MODULE_PATH 0 "${KATANA_ROOT}/plugins/Src/cmake")

set(CMAKE_CXX_STANDARD 11)

find_package(Katana PATHS "${KATANA_ROOT}/plugin_apis/cmake" REQUIRED)

if (NOT MSVC)
    add_compile_options(-Wall)
endif ()

# Ensure we don't compile DSOs with undefined symbols.
if (CMAKE_SYSTEM_NAME MATCHES Linux)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined")
endif ()

if (MSVC)
    # Prevent Boost auto-linking.
    add_definitions(-DBOOST_ALL_NO_LIB)
endif ()

#-------------------------------------------------------------------------------
# Find dependencies.
if (EXISTS $ENV{REZ_ILMBASE_ROOT})
    set(OPENEXR_ROOT $ENV{REZ_OPENEXR_ROOT})
    find_package(OpenEXR)
endif ()

if (EXISTS $ENV{REZ_BOOST_ROOT})
    set(BOOST_ROOT $ENV{REZ_BOOST_ROOT})
    find_package(Boost 1.55 COMPONENTS filesystem regex system thread)
endif ()

if (EXISTS $ENV{REZ_ZLIB_ROOT})
    set(ZLIP_ROOT $ENV{REZ_ZLIB_ROOT})
endif ()

if (EXISTS $ENV{REZ_TBB_ROOT})
    set(TBB_ROOT $ENV{REZ_TBB_ROOT})
    find_package(TBB)
endif ()

if (EXISTS $ENV{REZ_HDF5_ROOT})
    set(HDF5_ROOT $ENV{REZ_HDF5_ROOT})
    find_package(HDF5)
endif ()

if (EXISTS $ENV{REZ_ALEMBIC_ROOT})
    set(ALEMBIC_ROOT $ENV{REZ_ALEMBIC_ROOT})
    find_package(Alembic)
endif ()

find_package(Threads REQUIRED)
find_package(ZLIB)

# find_package(TinyXML)

#-------------------------------------------------------------------------------
# add projectss
add_subdirectory(DxUsdFeather)
