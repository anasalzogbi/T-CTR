#!/usr/bin/env bash
rootpath=/home/alzoghba/citeulike_data/terms_keywords_based
#for folder in 2k_20_P3_reduced 2k_20_P5_reduced 2k_20_P10_reduced 2k_30_P3_reduced 2k_30_P5_reduced 2k_30_P10_reduced 2k_40_P3_reduced 2k_40_P5_reduced 2k_40_P10_reduced
for folder in 2k_20_P3_reduced 2k_30_P3_reduced 2k_40_P3_reduced
do  
    screen -S lda_topics_extraction_${folder} -dm bash -c " source /home/alzoghba/HyPRec/py3.5_hyprec/bin/activate;  \
    python3 /home/alzoghba/IFUP2018/SciPRec_CTR/lib/lda_topics_gensim.py \
    -o ${rootpath}/${folder}/lda_topics \
    -f ${rootpath}/${folder}/mult.dat \
    -v ${rootpath}/${folder}/terms.dat \
    -k 200 250 \
    -c 5 ; exec sh;"    
done