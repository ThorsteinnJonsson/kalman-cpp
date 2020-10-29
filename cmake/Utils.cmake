#
# Print a message only if the `VERBOSE_OUTPUT` option is on
#

function(verbose_message content)
    if(${PROJECT_NAME}_VERBOSE_OUTPUT)
			message(STATUS ${content})
    endif()
endfunction()

#
# Add a target for formating the project using `clang-format` (i.e: cmake --build build --target clang-format)
#

function(add_clang_format_target)
    if(NOT ${PROJECT_NAME}_CLANG_FORMAT_BINARY)
			find_program(${PROJECT_NAME}_CLANG_FORMAT_BINARY clang-format)
    endif()

    if(${PROJECT_NAME}_CLANG_FORMAT_BINARY)
			if(${PROJECT_NAME}_BUILD_EXECUTABLE)
				add_custom_target(clang-format
						COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY}
						-i $${CMAKE_CURRENT_LIST_DIR}/${exe_sources} ${CMAKE_CURRENT_LIST_DIR}/${headers})
			elseif(${PROJECT_NAME}_BUILD_HEADERS_ONLY)
				add_custom_target(clang-format
						COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY}
						-i ${CMAKE_CURRENT_LIST_DIR}/${headers})
			else()
				add_custom_target(clang-format
						COMMAND ${${PROJECT_NAME}_CLANG_FORMAT_BINARY}
						-i ${CMAKE_CURRENT_LIST_DIR}/${sources} ${CMAKE_CURRENT_LIST_DIR}/${headers})
			endif()

			message(STATUS "Format the project using the `clang-format` target (i.e: cmake --build build --target clang-format).\n")
    endif()
endfunction()

#
# Link external library while ignoring warnings
#

function(target_link_libraries_system target)
  set(options PRIVATE PUBLIC INTERFACE)
  cmake_parse_arguments(TLLS "${options}" "" "" ${ARGN})
  foreach(op ${options})
    if(TLLS_${op})
      set(scope ${op})
    endif()
  endforeach(op)
  set(libs ${TLLS_UNPARSED_ARGUMENTS})

  foreach(lib ${libs})
    get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
    if(lib_include_dirs)
      if(scope)
        target_include_directories(${target} SYSTEM ${scope} ${lib_include_dirs})
      else()
        target_include_directories(${target} SYSTEM PRIVATE ${lib_include_dirs})
      endif()
    else()
      message("Warning: ${lib} doesn't set INTERFACE_INCLUDE_DIRECTORIES. No include_directories set.")
    endif()
    if(scope)
      target_link_libraries(${target} ${scope} ${lib})
    else()
      target_link_libraries(${target} ${lib})
    endif()
  endforeach()
endfunction(target_link_libraries_system)
