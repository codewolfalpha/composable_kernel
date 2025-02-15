include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

message(STATUS "Fetching GoogleTest")

list(APPEND GTEST_CMAKE_CXX_FLAGS
     -Wno-undef
     -Wno-reserved-identifier
     -Wno-global-constructors
     -Wno-missing-noreturn
     -Wno-disabled-macro-expansion
     -Wno-used-but-marked-unused
     -Wno-switch-enum
     -Wno-zero-as-null-pointer-constant
     -Wno-unused-member-function
     -Wno-comma
     -Wno-old-style-cast
     -Wno-deprecated
)
message(STATUS "Suppressing googltest warnings with flags: ${GTEST_CMAKE_CXX_FLAGS}")

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        b85864c64758dec007208e56af933fc3f52044ee
)

FetchContent_MakeAvailable(googletest)

target_compile_options(gtest PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gtest_main PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gmock PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gmock_main PRIVATE ${GTEST_CMAKE_CXX_FLAGS})

target_compile_definitions(gtest PUBLIC GTEST_HAS_SEH=0)
target_compile_definitions(gtest_main PUBLIC GTEST_HAS_SEH=0)
target_compile_definitions(gmock PUBLIC GTEST_HAS_SEH=0)
target_compile_definitions(gmock_main PUBLIC GTEST_HAS_SEH=0)

set_target_properties(gtest PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gtest_main PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gmock PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gmock_main PROPERTIES POSITION_INDEPENDENT_CODE ON)
