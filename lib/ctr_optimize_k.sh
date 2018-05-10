#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_data/terms_keywords_based/2k_10_P5_reduced
mkdir ${rootpath}/in-matrix-item_folds/logs
for k in 50 100 150 250
do
    mkdir ${rootpath}/in-matrix-item_folds/CTR_K_${k}/ 
    for fold in 1
    do
        mkdir ${rootpath}/in-matrix-item_folds/CTR_K_${k}/fold-${fold}
        screen -S CTR_k_${k}_fold_${fold} -dm bash -c " cd /home/alzoghba/ctr-blei; LD_LIBRARY_PATH=/home/alzoghba/gsl/lib; export LD_LIBRARY_PATH; \
        ./ctr-condor --save_lag 40 --directory ${rootpath}/in-matrix-item_folds/CTR_K_${k}/fold-${fold} \
        --user ${rootpath}/in-matrix-item_folds/fold-${fold}/train-fold_${fold}-users.dat \
        --item ${rootpath}/in-matrix-item_folds/fold-${fold}/train-fold_${fold}-items.dat \
        --mult ${rootpath}/mult.dat \
        --theta_init ${rootpath}/lda_topics/theta_${k}.dat \
        --beta_init ${rootpath}/lda_topics/beta_${k}.dat \
        --alpha_smooth 0.1 --random_seed 43 --num_factors ${k} > ${rootpath}/in-matrix-item_folds/logs/ctr_k_${k}_fold_${fold}.out; exec sh"
    done
done