# Empty list
SET(TCHEM_EXAMPLE_SOURCES "")

# Append examples that work for all device types
LIST(APPEND TCHEM_EXAMPLE_SOURCES
  TChem_KForwardReverseSurface.cpp
  TChem_GkSurfGas.cpp
  TChem_PlugFlowReactorRHS.cpp
  TChem_NetProductionRatePerMass.cpp
  TChem_NetProductionRateSurfacePerMass.cpp
  TChem_InitialCondSurface.cpp
  TChem_TransientContStirredTankReactor.cpp
  TChem_Jacobian.cpp
  TChem_IgnitionZeroD_ModelVariation.cpp
  TChem_IgnitionZeroDSA.cpp
  TChem_PlugFlowReactor.cpp
  TChem_PlugFlowReactorSmat.cpp
  TChem_SimpleSurface.cpp
  TChem_SurfaceRHS.cpp
  TChem_ThermalProperties.cpp
  TChem_JacobianSacado.cpp
  TChem_PlugFlowReactorJacobianSacado.cpp
  TChem_TransientContStirredTankReactorJacobianSacado.cpp
  TChem_JacobianVerification.cpp
  TChem_SourceTermToyProblem.cpp
  TChem_TransientContStirredTankReactorRHS.cpp
  TChem_Driver.cpp
  TChem_Driver_C.cpp
  TChem_PlugFlowReactorModelVariation.cpp
  TChem_TransientContStirredTankReactorModelVariation.cpp
  TChem_ConstantVolumeIgnitionReactorSmatrixSensitivityAnalysis.cpp
  TChem_IgnitionZeroD_SourceTermJacobian.cpp
  TChem_KForwardReverse.cpp
  TChem_KForwardReverseSurfaceModelVarition.cpp
  TChem_NetProductionRatePerMass_Yaml.cpp
  TChem_NetProductionRatePerMassNCAR.cpp
  TChem_ThermalProperties_Yaml.cpp
  TChem_NetProductionRateSurfacePerMass_Yaml.cpp
  TChem_RateOfProgressJacobian.cpp
  TChem_IgnitionZeroDJacobians.cpp
  TChem_AtmosphericChemistry.cpp
)


#
# Create examples for the above sources
#
FOREACH(TCHEM_EXAMPLE_SOURCE ${TCHEM_EXAMPLE_SOURCES} )
  STRING( REPLACE ".cpp" ".x" TCHEM_EXAMPLE_EXE ${TCHEM_EXAMPLE_SOURCE} )
  ADD_EXECUTABLE(${TCHEM_EXAMPLE_EXE} ${TCHEM_EXAMPLE_SOURCE})
  TARGET_LINK_LIBRARIES(${TCHEM_EXAMPLE_EXE} ${TCHEM_LINK_LIBRARIES})
  INSTALL(TARGETS ${TCHEM_EXAMPLE_EXE}
          PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
          DESTINATION "${CMAKE_INSTALL_PREFIX}/${TCHEM_INSTALL_EXAMPLE_PATH}")
ENDFOREACH()

# Fortran example
IF (TCHEM_ENABLE_FORTRAN)
  SET(TCHEM_EXAMPLE_F90_SOURCES "")

  # Append examples
  LIST(APPEND TCHEM_EXAMPLE_F90_SOURCES
    TChem_DriverNetProductionRatePerMass.F90
    TChem_DriverIgnitionZeroD.F90
  )

  ADD_LIBRARY(TChem_DriverModule TChem_DriverModule.F90)
  FOREACH(TCHEM_EXAMPLE_SOURCE ${TCHEM_EXAMPLE_F90_SOURCES} )
    STRING( REPLACE ".F90" ".x" TCHEM_EXAMPLE_EXE ${TCHEM_EXAMPLE_SOURCE} )
    ADD_EXECUTABLE(${TCHEM_EXAMPLE_EXE} ${TCHEM_EXAMPLE_SOURCE})
    TARGET_LINK_LIBRARIES(${TCHEM_EXAMPLE_EXE} TChem_DriverModule ${TCHEM_LINK_LIBRARIES})
    INSTALL(TARGETS ${TCHEM_EXAMPLE_EXE}
            PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
            DESTINATION "${CMAKE_INSTALL_PREFIX}/${TCHEM_INSTALL_EXAMPLE_PATH}")
  ENDFOREACH()
ENDIF()

#
# Testing data files
#
FILE(COPY data
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

INSTALL(DIRECTORY data
        DESTINATION ${CMAKE_INSTALL_PREFIX}/example)

#
FILE(COPY runs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

INSTALL(DIRECTORY runs
        DESTINATION ${CMAKE_INSTALL_PREFIX}/example)
