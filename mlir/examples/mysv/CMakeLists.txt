add_custom_target(MySV)
set_target_properties(MySV PROPERTIES FOLDER Examples)

set(LLVM_LINK_COMPONENTS
  Support
  )

include_directories(include/)

macro(add_mysv_chapter name)
  add_dependencies(Toy ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_mysv_chapter name)

add_mysv_chapter(mysv
  mysv.cpp
  AST.cpp
)

target_link_libraries(mysv
  PRIVATE
  MLIRSupport)
