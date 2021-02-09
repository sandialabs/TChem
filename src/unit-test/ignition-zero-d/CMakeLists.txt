#
# test
#
ADD_EXECUTABLE(tchem-test-ignition-zero-d.x TChem_Test_Driver.cpp)
TARGET_LINK_LIBRARIES(tchem-test-ignition-zero-d.x ${TCHEM_LINK_LIBRARIES})

#
# Testing method should be improved
#
FILE(COPY reference
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

INSTALL(DIRECTORY reference
        DESTINATION ${CMAKE_INSTALL_PREFIX}/unit-test/ignition-zero-d)

INSTALL(TARGETS tchem-test-ignition-zero-d.x
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${TCHEM_INSTALL_TEST_PATH}/ignition-zero-d")

ADD_TEST(TChemTest tchem-test-ignition-zero-d.x)
