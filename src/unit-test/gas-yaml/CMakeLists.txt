#
# test
#
ADD_EXECUTABLE(tchem-test-yaml.x TChem_Test_DriverYaml.cpp)
TARGET_LINK_LIBRARIES(tchem-test-yaml.x ${TCHEM_LINK_LIBRARIES})
#
# Testing method should be improved
#

INSTALL(TARGETS tchem-test-yaml.x
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${TCHEM_INSTALL_TEST_PATH}/gas-yaml")

ADD_TEST(TChemTest tchem-test-yaml.x)
