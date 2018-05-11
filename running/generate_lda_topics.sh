#!/usr/bin/env bash
rootpath=/vol1/data_anas/citeulike_2004_2007/time-based_split_out-of-matrix
for i in $(seq 6)
#for folder in 2k_40_P3_reduced 2k_40_P5_reduced 2k_40_P10_reduced
do
    folder=fold-${i}  
    screen -S lda_topics_extraction_${folder} -dm bash -c " source /home/alzoghba/HyPRec/py3.5_hyprec/bin/activate;  \
    python3 /home/alzoghba/SciPRec_CTR/lib/lda_topics_sklearn.py \
    -o ${rootpath}/${folder}/lda_sklearn \
    -f ${rootpath}/${folder}/mult.dat \
    -v ${rootpath}/${folder}/terms.csv \
    -k 150 ; exec sh;"    
done
