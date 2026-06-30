# FindNetCDF.cmake — locate a serial NetCDF C library.
#
# Tries, in order:
#   1. pkg-config (`netcdf`)
#   2. the netCDF CMake config package (`find_package(netCDF CONFIG)`)
#   3. plain header/library search
#
# Sets:
#   NetCDF_FOUND
#   NetCDF_INCLUDE_DIRS
#   NetCDF_LIBRARIES

if(NetCDF_FOUND)
    return()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_NETCDF QUIET netcdf)
endif()

# Fall back to the upstream CMake config package if pkg-config missed.
if(NOT PC_NETCDF_FOUND)
    find_package(netCDF CONFIG QUIET)
    if(netCDF_FOUND)
        set(NetCDF_INCLUDE_DIRS "${netCDF_INCLUDE_DIR}")
        set(NetCDF_LIBRARIES "${netCDF_LIBRARIES}")
        if(NOT NetCDF_LIBRARIES)
            set(NetCDF_LIBRARIES netCDF::netcdf)
        endif()
        set(NetCDF_FOUND TRUE)
    endif()
endif()

if(NOT NetCDF_FOUND)
    find_path(NetCDF_INCLUDE_DIRS
        NAMES netcdf.h
        HINTS ${PC_NETCDF_INCLUDE_DIRS} ${PC_NETCDF_INCLUDEDIR}
        PATH_SUFFIXES include)
    find_library(NetCDF_LIBRARIES
        NAMES netcdf
        HINTS ${PC_NETCDF_LIBRARY_DIRS} ${PC_NETCDF_LIBDIR}
        PATH_SUFFIXES lib lib64)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NetCDF
    REQUIRED_VARS NetCDF_LIBRARIES NetCDF_INCLUDE_DIRS)

mark_as_advanced(NetCDF_INCLUDE_DIRS NetCDF_LIBRARIES)
