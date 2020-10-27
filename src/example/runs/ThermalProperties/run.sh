
exec=$TCHEM_INSTALL_PATH/example/TChem_ThermalProperties.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
samplefile="sample.dat"

$exec --chemfile=$chemfile --thermfile=$thermfile --inputfile=$samplefile
