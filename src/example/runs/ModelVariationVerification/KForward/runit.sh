exec=$TCHEM_INSTALL_PATH/example/TChem_KForwardReverse.x
mech=$TCHEM_INSTALL_PATH/example/data/propane
chemfile=$mech/"chem.inp"
thermfile=$mech/"therm.dat"
samplefile=inputs/"sample.dat"
inputFileParamModifiers=inputs/"Factors.dat"

OMP_NUM_THREADS=8 OMP_PLACES=threads  OMP_PROC_BIND=spread $exec --chemfile=$chemfile --thermfile=$thermfile --samplefile=$samplefile --inputFileParamModifiers=$inputFileParamModifiers
