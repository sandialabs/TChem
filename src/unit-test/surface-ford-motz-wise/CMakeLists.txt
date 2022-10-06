#
# test
#
ADD_EXECUTABLE(tchem-test-surface-ford-motz-wise.x TChem_Test_Driver.cpp)
TARGET_LINK_LIBRARIES(tchem-test-surface-ford-motz-wise.x ${TCHEM_LINK_LIBRARIES})

#
# Testing method should be improved
#
FILE(COPY reference
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

INSTALL(DIRECTORY reference
        DESTINATION ${CMAKE_INSTALL_PREFIX}/unit-test/surface-ford-motz-wise)

INSTALL(TARGETS tchem-test-surface-ford-motz-wise.x 
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${TCHEM_INSTALL_TEST_PATH}/plug-flow-reactor")

ADD_TEST(TChemTest tchem-test-surface-ford-motz-wise.x)
