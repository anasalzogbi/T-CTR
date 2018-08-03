Database and Information Systems group  
University of Freiburg    
Germany  

Anas Alzogbi  
email: <alzoghba@informatik.uni-freiburg.de>  


# Temporal Collaborative Topic Regression (T-CTR)
Implementation for the paper [Time-aware Collaborative Topic Regression: Towards Higher Relevance in Textual Item Recommendation](http://ceur-ws.org/Vol-2132/paper2.pdf).  
Extends Collaborative Topic Modelling [CTR Recommender System](https://github.com/blei-lab/ctr) 
used in the work of [Wang and Blei](http://www.cs.princeton.edu/~chongw/papers/WangBlei2011.pdf) 
to consider the temporal aspect in recommendation.  
This project generates a recommender system model. To evaluate the generated model, 
use the [Recommender Evaluator System](https://github.com/anasalzogbi/Recommender_Evaluator), which is responsible for splitting the labeled data 
and computing the evaluation metrics.
 
### Rquirements:
- python 3.5
- numpy
### Dataset:
This work uses dataset collected from citeulike, you can downlowd it from [here](http://dbis.informatik.uni-freiburg.de/forschung/projekte/SciPRec/)
### Please cite using the following BibTex entry:
```
@inproceedings{DBLP:conf/sigir/Alzogbi18,
  author    = {Anas Alzogbi},
  title     = {Time-aware Collaborative Topic Regression: Towards Higher Relevance
               in Textual Item Recommendation},
  booktitle = {Proceedings of the 3rd Joint Workshop on Bibliometric-enhanced Information
               Retrieval and Natural Language Processing for Digital Libraries {(BIRNDL}
               2018) co-located with the 41st International {ACM} {SIGIR} Conference
               on Research and Development in Information Retrieval {(SIGIR} 2018),
               Ann Arbor, USA, July 12, 2018.},
  pages     = {10--23},
  year      = {2018},
  crossref  = {DBLP:conf/sigir/2018birndl},
  url       = {http://ceur-ws.org/Vol-2132/paper2.pdf},
  timestamp = {Mon, 09 Jul 2018 18:23:12 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/sigir/Alzogbi18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
### Directory structure:  
The following directory tree illustrates how the structure of data directory will be:   
|__ `[USER_RATINGS_FILE.dat]`  
|__ `[SPLIT_DIRECTORY]`  
......|__ fold`[1-5]`  
......|......|_ train-fold_`[1-5]`-items.dat  
......|......|_ test-fold_`[1-5]`-items.dat  
......|......|_ train-fold_`[1-5]`-users.dat  
......|......|_ test-fold_`[1-5]`-users.dat  
......|__ `[EXPERIMENT_DIRECTORY]`   
.............|_ `[EXPERIMENT_NAME]`\_eval_results.txt  
.............|_ results-matrix.npy  
.............|_ fold`[1-5]`  
.............|......|_ final-U.dat  
.............|......|_ final-V.dat  
.............|......|_ score.npy  
.............|......|_ results-users.dat   
