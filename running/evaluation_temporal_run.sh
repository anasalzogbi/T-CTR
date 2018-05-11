#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_2004_2007

# folds=( 1 2 3 4 6 )
# nu=( 173 484 978 1518 2716 )
folds=( 2 )
nu=( 484 ) 

for exp in Factor_adaptive_IE2_temporal_CTR_sklearn_K_150_V Factor_0.1_IE2_temporal_CTR_sklearn_K_150_V Factor_0.5_IE2_temporal_CTR_sklearn_K_150_V Factor_1_IE2_temporal_CTR_sklearn_K_150_V 
do
    for v in 1 10
    do  
	for ((i=0;i<${#folds[@]};++i)); do
        	screen -S temporal_eval_${exp}_${v}_fold${folds[i]} -dm bash -c " source /home/alzoghba/HyPRec/py3.5_hyprec/bin/activate;  \
		python3 /home/alzoghba/Recommender_evaluator/lib/evaluator.py -s -nu ${nu[i]} \
		-es ${rootpath}/time-based_split_out-of-matrix/fold-${folds[i]} \
		-x  ${rootpath}/time-based_split_out-of-matrix/${exp}_${v}/fold-${folds[i]}; exec sh;"
    	done
    done
done
