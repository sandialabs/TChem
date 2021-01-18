
exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD_ModelVariation.x
inputs=inputs/
save=1
dtmin=1e-12
dtmax=1e-2
tend=1.0
max_time_iterations=500
max_newton_iterations=20
atol_newton=1e-12
rtol_newton=1e-8
tol_time=1e-7

$exec --inputsPath=$inputs --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
