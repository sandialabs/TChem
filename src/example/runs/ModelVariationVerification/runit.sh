exec=$TCHEM_INSTALL_PATH/example/TChem_KForwardReverse.x

chemfile=inputs/"ThinkMech_chem.inp"
thermfile=inputs/"ThinkMech_therm.dat"
samplefile=inputs/"sample.dat"
inputFileParamModifiers=inputs/"Factors.dat"

OMP_NUM_THREADS=8 OMP_PLACES=threads  OMP_PROC_BIND=spread $exec --chemfile=$chemfile --thermfile=$thermfile --samplefile=$samplefile --inputFileParamModifiers=$inputFileParamModifiers
