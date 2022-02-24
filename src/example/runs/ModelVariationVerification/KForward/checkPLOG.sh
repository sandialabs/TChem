exec=$TCHEM_INSTALL_PATH/example/TChem_checkPLOGreactions.x
mech=$TCHEM_INSTALL_PATH/example/data/propane
chemfile=$mech/"chem.inp"
thermfile=$mech/"therm.dat"

run_this="$exec --chemfile=$chemfile \
                --thermfile=$thermfile \
                --initial-temperature=300 \
                --final-temperature=400 \
                --initial-pressure=100 \
                --final-pressure=4000000 \
                --npoints-temperature=100 \
                --npoints-pressure=100 \
                --factor-pre-exponential=100 \
                --verbose=true "
echo $run_this
eval $run_this
