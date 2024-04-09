function(set_project_warnings project_name)
	option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors." FALSE)
  set(GCC_WARNINGS
      -Wall
      -Wextra
      -Wshadow # warn user if a variable is beeing shadowed
      -Wnon-virtual-dtor # warn if a virtual class has a non virtual destructor
      -Wold-style-cast # warn if for old c-style casts
      -Wcast-align # warn for potentially performance reducing casts
      -Wunused # warn on anything being unused
      -Woverloaded-virtual # Warn if you overload (not overwrite) a virtual
                           # function
      -Wpedantic # warn if non-standard C++ is used
      -Wconversion # warn on type conversion that might loose data
      -Wsign-conversion # warn on sign conversion
      -Wnull-dereference # warn if a null dereference is detected
      -Wdouble-promotion # warn if a float is implicitly promoted to double
      -Wformat=2 # warn security issues around functions that format output
      -Wmisleading-indentation # warn if indentation implies blocks where thera
                               # re none
      -Wduplicated-cond # warn if if/else chain has duplicated conditions
      -Wduplicated-branches # warn if if/else branches have duplicated code
      -Wlogical-op # warn if logical ops are used when bitwise ops are propably
                   # wanted
      -Wuseless-cast # warn if you preform a cast to the same type
  )
  if(WARNINGS_AS_ERRORS)
    set(GCC_WARNINGS ${GCC_WARNINGS} -Werror)
  endif()

  target_compile_options(${project_name} INTERFACE ${GCC_WARNINGS})
endfunction()
