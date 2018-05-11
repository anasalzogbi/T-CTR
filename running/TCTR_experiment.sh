#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_2004_2007
    mkdir ${rootpath}/time-based_split_out-of-matrix/logs
    for k in 150 
    do
        for v in 1 10   
        do
	  for factor in adaptive 0.1 0.5 1 
	  do
            mkdir ${rootpath}/time-based_split_out-of-matrix/Factor_${factor}_IE2_temporal_CTR_sklearn_K_${k}_V_${v}/ 
            for fold in 2 
            do
                mkdir ${rootpath}/time-based_split_out-of-matrix/Factor_${factor}_IE2_temporal_CTR_sklearn_K_${k}_V_${v}/fold-${fold}
                screen -S Factor_${factor}_IE2_temporal_CTR_K_${k}_V_${v}_sklearn_fold_${fold} -dm bash -c "echo I_started; cd /home/alzoghba/temporal-ctr-blei; LD_LIBRARY_PATH=/home/alzoghba/gsl/lib; export LD_LIBRARY_PATH; \
		./ctr-condor --save_lag 40 \
                --user ${rootpath}/time-based_split_out-of-matrix/fold-${fold}/train-users.dat \
		--user_a ${rootpath}/time-based_split_out-of-matrix/fold-${fold}/train-users-weights_Factor_${factor}_IE2.dat \
		--item ${rootpath}/time-based_split_out-of-matrix/fold-${fold}/train-items.dat \
		--item_a ${rootpath}/time-based_split_out-of-matrix/fold-${fold}/train-items-weights_Factor_${factor}_IE2.dat \
		--directory ${rootpath}/time-based_split_out-of-matrix/Factor_${factor}_IE2_temporal_CTR_sklearn_K_${k}_V_${v}/fold-${fold} \
                --mult ${rootpath}/time-based_split_out-of-matrix/fold-${fold}/mult.dat \
                --theta_init ${rootpath}/time-based_split_out-of-matrix/fold-${fold}/lda_sklearn/theta_${k}.dat \
                --beta_init ${rootpath}/time-based_split_out-of-matrix/fold-${fold}/lda_sklearn/beta_${k}.dat \
                --lambda_v ${v} \
                --random_seed 43 --num_factors ${k} > ${rootpath}/time-based_split_out-of-matrix/logs/factor${factor}_IE2_temporal_ctr_k_${k}_v_${v}_sklearn_fold_${fold}.out; echo I_finished; exec sh;"
            done
          done
        done
    done

