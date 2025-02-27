if(${PROJECT_NAME}_ENABLE_CONAN)
  #
  # Setup Conan requires and options here:
  #

  set(PROJECT_CONAN_REQUIRES "eigen/3.3.7")
  set(PROJECT_CONAN_OPTIONS "")

  #
  # If `conan.cmake` (from https://github.com/conan-io/cmake-conan) does not exist, download it.
  #
  if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(
      STATUS
        "Downloading conan.cmake from https://github.com/conan-io/cmake-conan..."
    )
    file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
      "${CMAKE_BINARY_DIR}/conan.cmake"
    )
    message(STATUS "Cmake-Conan downloaded succesfully.")
  endif()

  include(${CMAKE_BINARY_DIR}/conan.cmake)

  conan_add_remote(NAME bincrafters 
      URL
          https://api.bintray.com/conan/bincrafters/public-conan
  )

  conan_cmake_run(
    REQUIRES
      ${PROJECT_CONAN_REQUIRES}
    OPTIONS
      ${PROJECT_CONAN_OPTIONS}
    BASIC_SETUP
      CMAKE_TARGETS # Individual targets to link to
    BUILD
      missing
  )

  verbose_message("Conan is setup and all requires have been installed.")
endif()
