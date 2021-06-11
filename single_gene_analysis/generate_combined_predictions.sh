#!/usr/bin/env bash


for model_class in '3SAA' '3SAB' '3SAC' '3SBA' '3SBB' '3SBC' '3SCA' '3SCB' '3SCC'
do

    OUT_PATH="predictions/joint_fit/"
    if [ ! -d ${OUT_PATH} ]
    then
      mkdir -p ${OUT_PATH}
    fi

    python predict_combined_model_marginals.py \
    fit_file=opt_results/joint_fit/${model_class}_best_joint_fit.npz \
    model=${model_class} \
    output_file=${OUT_PATH}/${model_class}_predictions.npz
done
