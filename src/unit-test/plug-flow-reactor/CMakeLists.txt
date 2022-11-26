#
# test
#
ADD_EXECUTABLE(tchem-test-plug-flow-reactor.x TChem_Test_Driver.cpp)
TARGET_LINK_LIBRARIES(tchem-test-plug-flow-reactor.x ${TCHEM_LINK_LIBRARIES})

#
# Testing method should be improved
#
FILE(COPY reference
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(TCHEM_ENABLE_INSTALL)
    INSTALL(DIRECTORY reference
            DESTINATION ${CMAKE_INSTALL_PREFIX}/unit-test/plug-flow-reactor)

    INSTALL(TARGETS tchem-test-plug-flow-reactor.x
            PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
            DESTINATION "${CMAKE_INSTALL_PREFIX}/${TCHEM_INSTALL_TEST_PATH}/plug-flow-reactor")
 ENDIF()

ADD_TEST(TChemTest tchem-test-plug-flow-reactor.x)
