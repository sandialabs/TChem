exec=$TCHEM_INSTALL_PATH/example/TChem_NetProductionRatePerMass_Yaml.x
inputs="inputs"
chemfile=$inputs/"gri30.yaml"
inputfile=$inputs/"sample.dat"

$exec --chemfile=$chemfile --inputfile=$inputfile
