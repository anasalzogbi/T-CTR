#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_2004_2007
    mkdir ${rootpath}/in-matrix-item_folds/logs
    for k in 400 
    do
        for v in 0.1
        do
            mkdir ${rootpath}/in-matrix-item_folds/CTR_sklearn_K_${k}_V_${v}/ 
            for fold in 1 2 3 4 5
            do
                mkdir ${rootpath}/in-matrix-item_folds/CTR_sklearn_K_${k}_V_${v}/fold-${fold}
                screen -S CTR_K_${k}_V_${v}_sklearn_fold_${fold} -dm bash -c "echo I_started; cd /home/alzoghba/ctr-blei; LD_LIBRARY_PATH=/home/alzoghba/gsl/lib; export LD_LIBRARY_PATH; \
                ./ctr-condor --save_lag 40 --directory ${rootpath}/in-matrix-item_folds/CTR_sklearn_K_${k}_V_${v}/fold-${fold} \
                --user ${rootpath}/in-matrix-item_folds/fold-${fold}/train-fold_${fold}-users.dat \
                --item ${rootpath}/in-matrix-item_folds/fold-${fold}/train-fold_${fold}-items.dat \
                --mult ${rootpath}/mult.dat \
                --theta_init ${rootpath}/lda_sklearn/theta_${k}.dat \
                --beta_init ${rootpath}/lda_sklearn/beta_${k}.dat \
                --lambda_v ${v} \
                --random_seed 43 --num_factors ${k} > ${rootpath}/in-matrix-item_folds/logs/ctr_k_${k}_v_${v}_sklearn_fold_${fold}.out; echo I_finished;"
            done
        done
    done

