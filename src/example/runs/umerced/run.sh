exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD_SourceTermJacobian.x
#inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/
inputs=H2/

chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
$exec --chemfile=$chemfile --thermfile=$thermfile 
