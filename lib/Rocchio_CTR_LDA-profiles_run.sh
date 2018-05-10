#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_data/terms_keywords_based
#for folder in 2k_10_P3_reduced 2k_10_P5_reduced 2k_10_P10_reduced 2k_20_P3_reduced 2k_20_P5_reduced 2k_20_P10_reduced 2k_30_P3_reduced 2k_30_P5_reduced 2k_30_P10_reduced 2k_40_P3_reduced 2k_40_P5_reduced
#for folder in 2k_40_P5_reduced 2k_40_P10_reduced 2k_50_P3_reduced 2k_50_P5_reduced 2k_100_P3_reduced 2k_100_P5_reduced
for folder in 2k_30_P3_reduced
do
    line=$(head -n 1 ${rootpath}/${folder}/stats.txt)
    IFS='(' read -a v <<< "${line}"
    v=${v[1]//)/}
    v=${v// /}
    IFS=',' read -a v <<< "${v}"
    users=${v[0]}
    items=${v[1]}
    echo $users
    echo $items
    for k in 200
    do
        screen -S Rocchio_LDAPROFILE_Cosine_${folder}_k_${k} -dm bash -c " source /home/alzoghba/HyPRec/py3.5_hyprec/bin/activate;  \
        python3 /home/alzoghba/IFUP2018/SciPRec_CTR/lib/rocchio_based_recommender.py \
        -u ${rootpath}/${folder}/in-matrix-item_folds \
        -p ${rootpath}/${folder}/lda_topics/theta_${k}.dat \
        -ps ${rootpath}/${folder}/in-matrix-item_folds/CTR_K_${k}/fold-1/final-V.dat ${rootpath}/${folder}/in-matrix-item_folds/CTR_K_${k}/fold-2/final-V.dat ${rootpath}/${folder}/in-matrix-item_folds/CTR_K_${k}/fold-3/final-V.dat ${rootpath}/${folder}/in-matrix-item_folds/CTR_K_${k}/fold-4/final-V.dat ${rootpath}/${folder}/in-matrix-item_folds/CTR_K_${k}/fold-5/final-V.dat \
        -s cosine \
        -e CTR_Rocchio_LDAPROFILE_Cosine_k_${k} \
        -splt ${rootpath}/${folder}/in-matrix-item_folds \
        -n ${users} -m ${items} -ldau -l 500 ;"
    done
done