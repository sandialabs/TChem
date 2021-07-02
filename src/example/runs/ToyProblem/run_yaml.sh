exec=$TCHEM_INSTALL_PATH/example/TChem_SourceTermToyProblem.x
chemfile="inputs/chem.yaml"
outputfile="omega_yaml.dat"
$exec --chemfile=$chemfile --outputfile=$outputfile --useYaml=true
