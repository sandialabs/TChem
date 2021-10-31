
exec=$TCHEM_INSTALL_PATH/example/TChem_SimpleSurface.x
mech=$TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas

run_this="$exec --chemfile=$mech/chem.inp \
                --thermfile=$mech/therm.dat \
                --samplefile=inputs/sample.dat \
                --surf-chemfile=$mech/chemSurf.inp \
                --surf-thermfile=$mech/thermSurf.dat \
                --surf-inputfile=inputs/inputSurf.dat \
                --max-newton-iterations=20 \
                --atol-newton=1e-18 \
                --rtol-newton=1e-6 \
                --tol-time=1e-10 \
                --dtmin=1e-20 \
                --dtmax=1e-8 \
                --tend=1e-3 \
                --time-iterations-per-interval=10 \
                --max-time-iterations=12000 "

echo $run_this
eval $run_this
