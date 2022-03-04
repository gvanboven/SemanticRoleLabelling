# Semantic Role Labelling

In this repository, our code can be found to implement a Semantic Role Labelling System.   
We have created this code for the second assignment for the course NLP Technologies at the Vrije Universiteit Amsterdam.  
We train our system on data from the Universal Proposition Banks dataset.   
This readme consists of two parts. In the first part we present a description of the code and how to run it. In the second part we describe our task and present our results.  

----------------------
## Repository description

This repository consists of 3 folders, which we will each describe. 

#### `.\data`
This folder contains the train, dev and test data we use for our model.

#### `.\models`
This folder contains the W2V word embeddings model we use to encode tokens during training. This model is currently excluded because of its large size.    
The model should thus be downloaded and placed in this folder in order for the code to work. The model can be downloaded from https://code.google.com/archive/p/word2vec/ 

#### `.\code`

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

----------------------
`feature_extraction.py` can be used to extract features from a Universal Proposition Banks dataset SRL conll file. The code carries out two steps :   
i. It restructes the data. In the original datasets, tokens have their own rows, sentences are separated by empty lines, and predicates and their arguments are presented in columns. 
In the restructured version of the data, the empty lines are taken out, and each token-predicate combination gets its own row. This means that if a sentence contains 10 tokens and 3 predicates, this sentence will be represented over 10 * 3 = 30 rows.     
ii. It extracts additional features for the tokens and the predicates, and adds these as columns. More specifically, we extract the following features:  

* `sent_index` - the index of the sentence
* `token_index` - the index of the token 
* `token` - the token itself
* `head_lemma` - the lemma of the head of the token 
* `predicate_descendant` - binary feature, whether the token is a descendant of the predicate or not
* `path_to_predicate_length`- the length of the dependency path from the token to the predicate. If there is no path, the value is 100
* `lemma` - the lemma of the token, extracted by Spacy
* `pos` - the POS of the token, extracted by Spacy
* `postag` - the POStag of the token, extracted by Spacy
* `prev_token` - the preceding token, extracted by Spacy
* `prev_pos` - the POS of the preceding token, extracted by Spacy
* `dependency` - the syntactic dependency, extracted by Spacy
* `head_text` - the lemma of the head, extracted by Spacy
* `predicate_lemma` - the lemma of the predicate
* `predicate_index` - the index of the predicate
* `predicate_pos` - the POS of the predicate, extracted by Spacy
* `predicate_postag` -  the POStag of the predicate, extracted by Spacy
* `predicate_dependency` -  the syntactic dependency, extracted by Spacy
* `argument` - the (gold) argument of the token for the current predicate

This file can be run as follows:    
`python .\feature_extraction.py '[conll_input_file_path]' '[conll_output_file_path]'`    
as for instance      
`python feature_extraction.py '../data/srl_univprop_en.dev.conll' '../data/srl_univprop_en.dev_features.conll'` 
where `[conll_input_file_path]` is the original conll data file, and `[conll_output_file_path]` is the path where the extracted data will be saved.


----------------------
`predicate_extraction.py`

TODO

----------------------
`argument_prediction.py`

This file creates and trains an SVM on the training data, makes predictions on the test data and saves the predictions in an output file.

This file can be run as follows:    
`python .\feature_extraction.py '[conll_train_file_path]' '[conll_test_file_path]' '[conll_output_file_path]' [number_of_included_rows] [selected features]`    
as for instance 
`python argument_prediction.py '../data/srl_univprop_en.train_features.conll' '../data/srl_univprop_en.test_features.conll' '../data/srl_univprop_en.test_predictions.conll'  1000`
where `[conll_train_file_path]` is the conll training data, `[conll_test_file_path]` is the conll test data, and `[conll_output_file_path]` is the path to where the predictions will be saved.
`[number_of_included_rows]` indicated the number of rows from that should be included in training and testing. This should be an integer. If all rows should be included in training and testing, the input for this argument should be `'all'`. The reason we included this argument is because the training dataset is very large and when using this dataset for training the argument prediction model it might take very long to train, or cause memory errors.    
Finally, there is an optional feature `[selected features]`, which to allows to select a subset of the features we extract to be included in argument prediction TODO 

----------------------
`evaluate.py`

This file evaluates machine predictions in a conll file wich contains both gold labels and machine annotations. 
This file prints out the confusion matrix and the overall macro precision, recall and F1, as well as the micro F1 score.
Aditionally, the precision, recall and F1 are computed per class, and a table is printed out that gives an overview of these scores.

This file can be run as follows:    
`python .\feature_extraction.py '[conll_input_file_path]' [gold_index] [predictions_index]`    
as for instance 
`python evaluate.py '../data/SRL_argument_test_predictions.conll' -2 -1`
Where `'[conll_input_file_path]'` is the file  that contains the gold labels and the predictions, `[gold_index]` the index of the column in which the gold annotations can be found, and `[predictions_index]` is the index of the column that contains the machine predictions.

----------------------
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
