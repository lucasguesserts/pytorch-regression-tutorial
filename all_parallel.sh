export EXE="./curve_fit_architecture_study.py"
export OUTPUT_DIR="output/"
export LOG_FILE="curve_fit.log"
export NUMBER_OF_THREADS=$(($(nproc --all) / 3))

mkdir -p $OUTPUT_DIR

echo "number of hidden layers,number of nodes in each hidden layer,number of trainable paramters,iterations,learning rate,mean square error" > $LOG_FILE

solve () {
    local NL=$1
    local NN=$2
    local LR=$3
    local I=$4
    local FIG="${OUTPUT_DIR}/exp_nl_${NL}_nn_${NN}_lr_${LR}_i_${I}.jpg"
    # echo "python $EXE -nl=$NL -nn=$NN -lr=$LR -i=$I -f=$FIG --csv"
    python $EXE -nl=$NL -nn=$NN -lr=$LR -i=$I -f=$FIG --csv
}
export -f solve

parallel \
    -j $NUMBER_OF_THREADS \
    --verbose \
    --keep-order \
    solve \
    ::: 2 3 4 \
    ::: 4 8 16 32 64 \
    ::: 1.0e-0 1.0e-1 1.0e-2 1.0e-3 1.0e-4 \
    ::: 16000 \
    >> $LOG_FILE
