#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_data/terms_keywords_based
#for dataFolder in 2k_20_P3_reduced 2k_20_P5_reduced 2k_20_P10_reduced 2k_30_P3_reduced 2k_30_P5_reduced 2k_30_P10_reduced 2k_40_P3_reduced 2k_40_P5_reduced
for dataFolder in 2k_20_P3_reduced 
do
    mkdir ${rootpath}/${dataFolder}/in-matrix-item_folds/logs
    for k in 200
    do
        for v in 500 1000
        do
            mkdir ${rootpath}/${dataFolder}/in-matrix-item_folds/CTR_K_${k}_V_${v}/ 
            for fold in 1 2 3 4 5
            do
                mkdir ${rootpath}/${dataFolder}/in-matrix-item_folds/CTR_K_${k}_V_${v}/fold-${fold}
                screen -S CTR_K_${k}_V_${v}_fold_${fold}_${dataFolder} -dm bash -c " cd /home/alzoghba/ctr-blei; LD_LIBRARY_PATH=/home/alzoghba/gsl/lib; export LD_LIBRARY_PATH; \
                ./ctr-condor --save_lag 40 --directory ${rootpath}/${dataFolder}/in-matrix-item_folds/CTR_K_${k}_V_${v}/fold-${fold} \
                --user ${rootpath}/${dataFolder}/in-matrix-item_folds/fold-${fold}/train-fold_${fold}-users.dat \
                --item ${rootpath}/${dataFolder}/in-matrix-item_folds/fold-${fold}/train-fold_${fold}-items.dat \
                --mult ${rootpath}/${dataFolder}/mult.dat \
                --theta_init ${rootpath}/${dataFolder}/lda_topics/theta_${k}.dat \
                --beta_init ${rootpath}/${dataFolder}/lda_topics/beta_${k}.dat \
                --lambda_v ${v} \
                --alpha_smooth 0.1 --random_seed 43 --num_factors ${k} > ${rootpath}/${dataFolder}/in-matrix-item_folds/logs/ctr_k_${k}_v_${v}_fold_${fold}.out; "
            done
        done
    done
done