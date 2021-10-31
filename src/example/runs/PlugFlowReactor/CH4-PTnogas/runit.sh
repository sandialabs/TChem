
exec=$TCHEM_INSTALL_PATH/example/TChem_PlugFlowReactor.x
inputs=$TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas/

run_this="$exec --inputs-path=$inputs \
                --initial-condition=false \
                --use-prefix-path=true \
                --transient-initial-condition=true \
                --reactor-area=0.00053 \
                --catalytic-perimeter=0.025977239243415308 \
                --max-newton-iterations=20 \
                --atol-newton=1e-18 \
                --rtol-newton=1e-8 \
                --tol-z=1e-8 \
                --dzmin=1e-20 \
                --dzmax=1e-3 \
                --zend=0.025 \
                --max-z-iterations=310 "

echo $run_this
eval $run_this
