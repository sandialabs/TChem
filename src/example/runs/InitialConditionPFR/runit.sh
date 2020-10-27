
exec=$TCHEM_INSTALL_PATH/example/TChem_SimpleSurface.x
inputs=inputs/
save=1
dtmin=1e-12
dtmax=1e-8
tend=1e-3
max_newton_iterations=20
max_time_iterations=6000
atol_newton=1e-12
rtol_newton=1e-8
tol_time=1e-10
 
$exec --prefixPath=$inputs --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations

