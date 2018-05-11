#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_2004_2007

folds=( 6 )
nu=( 2716 )

for exp in CTR_sklearn_K_150_V         
do
    for v in 0.001 
    do  
	for ((i=0;i<${#folds[@]};++i)); do
        	screen -S temporal_eval_${exp}_${v}_fold${folds[i]} -dm bash -c " source /home/alzoghba/HyPRec/py3.5_hyprec/bin/activate;  \
		python3 /home/alzoghba/Recommender_evaluator/lib/evaluator.py -s -nu ${nu[i]} \
		-es ${rootpath}/time-based_split_out-of-matrix/fold-${folds[i]} \
		-x  ${rootpath}/time-based_split_out-of-matrix/${exp}_${v}/fold-${folds[i]}; exec sh;"
    	done
    done
done
