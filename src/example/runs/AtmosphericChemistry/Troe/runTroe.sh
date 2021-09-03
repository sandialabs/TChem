exec=$TCHEM_INSTALL_PATH/example/TChem_NetProductionRatePerMassNCAR.x
chemfile="config_troe/config_troe.yaml"

$exec --chemfile=$chemfile --unit-test="troe"
