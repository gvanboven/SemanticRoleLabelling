# Semantic Role Labelling
In this repository, our code can be found to implement a Semantic Role Labelling System.   
We have created this code for the second assignment for the course NLP Technologies at the Vrije Universiteit Amsterdam.  
We train our system on data from the Universal Proposition Banks dataset.   
This readme consists of two parts. In the first part we present a description of the code and how to run it. In the second part we describe our task and present our results.  
 ----------------------
## Repositoty description

This repository consists of 3 folders, which we will each describe. 

### `.\data`
This folder contains the train, dev and test data we use for our model.

### `.\models`
This folder contains the W2V word embeddings model we use to encode tokens during training. This model is currently excluded because of its large size.    
The model should thus be downloaded and placed in this folder in order for the code to work. The model can be downloaded from https://code.google.com/archive/p/word2vec/ 

### `.\code`

This folder contains our code. We will describe each file   

`main.py` can be used to train the entire system. It carries out the following steps:  
i. it extracts features for SRL predicate and argument classification and saves the features in a new conll file. This new dataset will be named `[conll_train_file_path]_features.conll`   
ii. extracts the predicates of the test sentences using a rule-based method, and evaluates the performance     
iii. trains a SVM classifier to predict the agruments, uses this model to make predictions on the test data and evaluates the predictions    

For both evaluations, the results (in terms of precision, recall and F1 score) will be printed, both per class and for the entire dataset (both macro and micro average).

This file can be called as follows:    
`python .\main.py '[conll_train_file_path]', '[conll_test_file_path]' [number_of_included_rows] [selected features]` 
as for example:
`'python main.py '../data/srl_univprop_en.train.conll' '../data/srl_univprop_en.dev.conll' 100'`
Here the [conll_train_file_path] is the path to the conll training data, and [conll_test_file_path] is the path to the conll test data.   
`[number_of_included_rows]` indicated the number of rows from that should be included in training and testing. This should be an integer. If all rows should be included in training and testing, the input for this argument should be `'all'`. The reason we included this argument is because the training dataset is very large and when using this dataset for training the argument prediction model it might take very long to train, or cause memory errors.    
Finally, there is an optional feature `[selected features]`, which to allows to select a subset of the features we extract to be included in argument prediction TODO 

## Task description and results

### Predicate extraction

#### Task description

#### Results


### Argument classification

#### Task description

#### Features

##### Description

##### Feature extraction

#### Machine Learning

#### Results
