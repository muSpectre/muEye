# FindNetCDF.cmake — locate a serial NetCDF C library.
#
# Tries, in order:
#   1. pkg-config (`netcdf`)                      — Linux / macOS (Homebrew, apt)
#   2. the netCDF CMake config package (CONFIG)   — Windows / vcpkg / conda
#   3. a plain header/library search
#
# Sets:
#   NetCDF_FOUND
#   NetCDF_INCLUDE_DIRS   (may be empty when using an imported target that
#                          carries its include directories transitively)
#   NetCDF_LIBRARIES      (a path, a list, or the imported target netCDF::netcdf)

if(NetCDF_FOUND)
    return()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_NETCDF QUIET netcdf)
endif()

if(PC_NETCDF_FOUND)
    set(NetCDF_INCLUDE_DIRS "${PC_NETCDF_INCLUDE_DIRS}")
    if(PC_NETCDF_LINK_LIBRARIES)
        set(NetCDF_LIBRARIES "${PC_NETCDF_LINK_LIBRARIES}")
    else()
        set(NetCDF_LIBRARIES "${PC_NETCDF_LIBRARIES}")
    endif()
    set(NetCDF_FOUND TRUE)
endif()

# Fall back to the upstream/vcpkg CMake config package.
if(NOT NetCDF_FOUND)
    find_package(netCDF CONFIG QUIET)
    if(netCDF_FOUND)
        if(netCDF_LIBRARIES)
            set(NetCDF_LIBRARIES "${netCDF_LIBRARIES}")
        else()
            set(NetCDF_LIBRARIES netCDF::netcdf)
        endif()
        # Prefer the config's include var, else pull it off the imported target.
        if(netCDF_INCLUDE_DIR)
            set(NetCDF_INCLUDE_DIRS "${netCDF_INCLUDE_DIR}")
        elseif(TARGET netCDF::netcdf)
            get_target_property(_nc_inc netCDF::netcdf
                INTERFACE_INCLUDE_DIRECTORIES)
            if(_nc_inc)
                set(NetCDF_INCLUDE_DIRS "${_nc_inc}")
            endif()
        endif()
        set(NetCDF_FOUND TRUE)
    endif()
endif()

# Last resort: search the filesystem directly.
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
# Only the library is strictly required: when NetCDF is consumed as an imported
# target its include directories are propagated transitively, so an empty
# NetCDF_INCLUDE_DIRS is not a failure.
find_package_handle_standard_args(NetCDF REQUIRED_VARS NetCDF_LIBRARIES)

mark_as_advanced(NetCDF_INCLUDE_DIRS NetCDF_LIBRARIES)
