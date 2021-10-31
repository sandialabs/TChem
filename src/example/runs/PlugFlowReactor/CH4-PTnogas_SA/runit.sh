exec=$TCHEM_INSTALL_PATH/example/TChem_PlugFlowReactor.x
mech=$TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas

run_this="$exec --chemfile=$mech/chem.inp \
                --thermfile=$mech/therm.dat \
                --samplefile=inputs/sample.dat \
                --surf-chemfile=$mech/chemSurf.inp \
                --surf-thermfile=$mech/thermSurf.dat \
                --surf-inputfile=inputs/inputSurf.dat \
                --velocity-inputfile=inputs/inputVelocity.dat \
                --outputfile=PFRSolution.dat \
                --initial-condition=false \
                --transient-initial-condition=true \
                --reactor-area=0.00053 \
                --catalytic-perimeter=0.025977239243415308 \
                --max-newton-iterations=20 \
                --atol-newton=1e-16 \
                --rtol-newton=1e-6 \
                --tol-z=1e-6 \
                --dzmin=1e-30 \
                --dzmax=1e-3 \
                --zend=0.025 \
                --max-z-iterations=400 "

echo $run_this
eval $run_this
