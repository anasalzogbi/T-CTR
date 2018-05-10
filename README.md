Database and Information Systems group  
University of Freiburg    
Germany  

Anas Alzogbi  
email: <alzoghba@informatik.uni-freiburg.de>  


# T-CTR
Temporal Collaborative Topic Regression for recommendation. 
Extends Collaborative Topic Modelling [CTR Recommender System](https://github.com/blei-lab/ctr) 
used in the work of [Wang and Blei](http://www.cs.princeton.edu/~chongw/papers/WangBlei2011.pdf) 
to consider the temporal aspect in recommendation.  
This project generates a recommender system model. To evaluate the generated model, 
use the [Recommender Evaluator System](https://github.com/anasalzogbi/Recommender_Evaluator), which is responsible for splitting the labeled data 
and computing the evaluation metrics.
 
### Rquirements:
- python 3.5
- numpy


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
