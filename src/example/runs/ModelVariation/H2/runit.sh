exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD_ModelVariation.x
mech=$TCHEM_INSTALL_PATH/example/data/H2

run_this="$exec --chemfile=$mech/chem.inp \
                --thermfile=$mech/therm.dat \
                --samplefile=inputs/sample.dat \
                --input-file-param-modifiers=inputs/ParameterModifiers.dat \
                --tol-time=1e-7 \
                --dtmin=1e-20 \
                --dtmax=1e-2 \
                --tend=1.0 \
                --max-newton-iterations=20 \
                --atol-newton=1e-18 \
                --rtol-newton=1e-8 \
                --max-time-iterations=1000"

echo $run_this
eval $run_this
