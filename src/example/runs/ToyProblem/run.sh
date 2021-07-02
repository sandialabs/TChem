exec=$TCHEM_INSTALL_PATH/example/TChem_SourceTermToyProblem.x
chemfile="inputs/chem.inp"
thermfile="inputs/therm.dat"
outputfile="omega.dat"

$exec --chemfile=$chemfile --thermfile=$thermfile --outputfile=$outputfile  
