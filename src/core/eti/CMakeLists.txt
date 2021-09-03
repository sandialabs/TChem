SET(TCHEM_SLFAD_ETI_FILE TChem_IgnitionZeroD_SacadoJacobian;
                         TChem_TransientContStirredTankReactorSacadoJacobian;
                         TChem_PlugFlowReactorSacadoJacobian;
                         TChem_IgnitionZeroD
)

SET(TCHEM_SCALAR_ETI_FILE TChem_IgnitionZeroD
)

IF (Kokkos_ENABLE_CUDA)
  SET(TCHEM_EXEC_SPACE exec_space;
                       host_exec_space
  )
ELSE()
  SET(TCHEM_EXEC_SPACE host_exec_space)
ENDIF()

SET(TCHEM_SLFAD_SIZE 16;
                     32;
                     64;
                     128;
                     256;
                     512;
                     1024
)

FOREACH(ETI_FILE IN LISTS TCHEM_SLFAD_ETI_FILE)
  FOREACH(EXEC_SPACE IN LISTS TCHEM_EXEC_SPACE)
    IF (${EXEC_SPACE} STREQUAL "exec_space")
      SET(EXEC_SPACE_NAME Device)
    ELSEIF (${EXEC_SPACE} STREQUAL "host_exec_space")
      SET(EXEC_SPACE_NAME Host)
    ELSE()
      SET(EXEC_SPACE_NAME Unknown)
    ENDIF()

    FOREACH(SLFAD_SIZE IN LISTS TCHEM_SLFAD_SIZE)
      SET(VALUE_TYPE_NAME SLFad_${SLFAD_SIZE})
      MESSAGE(STATUS "Generating ETI: ${ETI_FILE}_${EXEC_SPACE_NAME}_${VALUE_TYPE_NAME}.cpp")
      CONFIGURE_FILE(${ETI_FILE}_ETI.in 
                     ${ETI_FILE}_${EXEC_SPACE_NAME}_${VALUE_TYPE_NAME}.cpp)
    ENDFOREACH()
  ENDFOREACH()
ENDFOREACH()

FOREACH(ETI_FILE IN LISTS TCHEM_SCALAR_ETI_FILE)
  FOREACH(EXEC_SPACE IN LISTS TCHEM_EXEC_SPACE)
    IF (${EXEC_SPACE} STREQUAL "exec_space")
      SET(EXEC_SPACE_NAME Device)
    ELSEIF (${EXEC_SPACE} STREQUAL "host_exec_space")
      SET(EXEC_SPACE_NAME Host)
    ELSE()
      SET(EXEC_SPACE_NAME Unknown)
    ENDIF()
    SET(SLFAD_SIZE 0)
    SET(VALUE_TYPE_NAME Scalar)
    MESSAGE(STATUS "Generating ETI: ${ETI_FILE}_${EXEC_SPACE_NAME}_${VALUE_TYPE_NAME}.cpp")
    CONFIGURE_FILE(${ETI_FILE}_ETI.in 
                   ${ETI_FILE}_${EXEC_SPACE_NAME}_${VALUE_TYPE_NAME}.cpp)
  ENDFOREACH()
ENDFOREACH()
