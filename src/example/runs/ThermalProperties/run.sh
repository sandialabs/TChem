
exec=$TCHEM_INSTALL_PATH/example/TChem_ThermalProperties.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/gri3.0/


run_this="$exec --chemfile=${inputs}chem.inp \
                --thermfile=${inputs}therm.dat \
                --inputfile=sample.dat "

echo $run_this
eval $run_this                
