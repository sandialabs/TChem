# Empty list
SET(TCHEM_EXAMPLE_SOURCES "")

# Append examples that work for all device types
LIST(APPEND TCHEM_EXAMPLE_SOURCES
  TChem_KForwardReverseSurface.cpp
  TChem_GkSurfGas.cpp
  TChem_NetProductionRateSurfacePerMass.cpp
  TChem_PlugFlowReactorRHS.cpp
  TChem_NetProductionRatePerMass.cpp
  TChem_Jacobian.cpp
  TChem_NewtonSolver.cpp
  TChem_TrBDF2.cpp
  TChem_IgnitionZeroD.cpp
  TChem_DenseUTV.cpp
  TChem_IgnitionZeroDSA.cpp
  TChem_PlugFlowReactor.cpp
  TChem_PlugFlowReactorSmat.cpp
  TChem_SimpleSurface.cpp
  TChem_SurfaceRHS.cpp
  TChem_ThermalProperties.cpp
  TChem_InitialCondSurface.cpp
  TChem_TransientContStirredTankReactor.cpp
  TChem_SourceTermToyProblem.cpp
  TChem_IgnitionZeroD_ModelVariation.cpp
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
