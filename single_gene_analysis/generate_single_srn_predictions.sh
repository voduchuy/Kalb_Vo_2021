#!/usr/bin/env bash


for model_class in "C3SB" #"2SA" "2SB" "3SA" "3SB" "3SC" "3SD"
do
for species in "tnfa" "il1b"
do
  for condition in "NoInhibitors"
  do
    OUT_PATH="predictions/single_cond_fits/"
    if [ ! -d ${OUT_PATH} ]
    then
      mkdir -p ${OUT_PATH}
    fi

    python predict_single_condition_marginals.py \
    fit_file=opt_results/single_cond_fits/bests/${model_class}_${species}_${condition}_best_1_fits.npz \
    model_class=${model_class} \
    output_file=${OUT_PATH}/predictions_${model_class}_${species}_${condition}.npz
  done
done
done
