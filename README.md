README.md
Wie heeft toegang
V
G
X
Systeemeigenschappen
Type
Tekst
Grootte
17 KB
Gebruikte opslag
17 KB
Locatie
SRL
Eigenaar
Vicky Kyrmanidi
Gewijzigd
8 mrt. 2022 door Vicky Kyrmanidi
Geopend
09:37 door mij
Gemaakt
8 mrt. 2022
Voeg een beschrijving toe
Kijkers kunnen downloaden
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

This folder contains our code. We will describe each file:   

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
Finally, there is an optional feature `[selected features]`, which to allows to select a subset of the features we extract to be included in argument prediction. The features should be writting as a string of selected features separated by spaces, as in the example above. If this is not defined, the following features will be used (see the description for `feature_extraction.py` for further details on these features): `token_index`, `token`, `head_lemma`, `predicate_descendant`, `path_to_predicate_length`,  `pos`, `postag`, `prev_token`, `prev_pos`, `dependency`, `head_text`, `predicate_lemma`, `predicate_index`, `predicate_pos`, `predicate_dependency`.

----------------------
`feature_extraction.py` can be used to extract features from a Universal Proposition Banks dataset SRL conll file. The code carries out two steps :   
i. It restructes the data. In the original datasets, tokens have their own rows, sentences are separated by empty lines, and predicates and their arguments are presented in columns.
In the restructured version of the data, the empty lines are taken out, and each token-predicate combination gets its own row. This means that if a sentence contains 10 tokens and 3 predicates, this sentence will be represented over 10 * 3 = 30 rows.     
ii. It extracts additional features for the tokens and the predicates, and adds these as columns. More specifically, we extract the following features:  

* `sent_index` - the index of the sentence
* `token_index` - the index of the token
* `token` - the token itself
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

This file performs the predicate prediction task using a rule-based and a machine learning method. It creates two new files for the processed train and test datasets in the same path as the original ones (The datasets need to be processed to serve the task). It prints the distribution of predicate features on which the rule-based approached is based. For the machine learning approach, it providesthe option of representing the tokens as word embeddings. For that the language model: `GoogleNews-vectors-negative300.bin.gz`, needs to be downloaded and loaded using gensim package. Finally, it prints the evaluation  of the rule based approach and the machine learning approach in a full classification report produced using the `sklearn` package.

This file and be run either within the main pipeline or separately. The independent execution needs to be run along with the path of the original train file and that of the test/dev file as follows:
- `python predicate_extraction.py '[path to original train file]' '[path to original test/dev file]'`
----------------------
`argument_prediction.py`

This file creates and trains an SVM on the training data, makes predictions on the test data and saves the predictions in an output file.

This file can be run as follows:    
`python .\feature_extraction.py '[conll_train_file_path]' '[conll_test_file_path]' '[conll_output_file_path]' [number_of_included_rows] [selected features]`    
as for instance
`python argument_prediction.py '../data/srl_univprop_en.train_features.conll' '../data/srl_univprop_en.test_features.conll' '../data/srl_univprop_en.test_predictions.conll'  1000  "token head_lemma predicate_lemma"`
where `[conll_train_file_path]` is the conll training data, `[conll_test_file_path]` is the conll test data, and `[conll_output_file_path]` is the path to where the predictions will be saved.
`[number_of_included_rows]` indicated the number of rows from that should be included in training and testing. This should be an integer. If all rows should be included in training and testing, the input for this argument should be `'all'`. The reason we included this argument is because the training dataset is very large and when using this dataset for training the argument prediction model it might take very long to train, or cause memory errors.    
Finally, there is an optional feature `[selected features]`, which to allows to select a subset of the features we extract to be included in argument prediction. The features should be writting as a string of selected features separated by spaces, as in the example above. If this is not defined, the following features will be used: `token_index`, `token`, `head_lemma`, `predicate_descendant`, `path_to_predicate_length`,  `pos`, `postag`, `prev_token`, `prev_pos`, `dependency`, `head_text`, `predicate_lemma`, `predicate_index`, `predicate_pos`, `predicate_dependency`.

----------------------
`evaluate.py`

This file evaluates machine predictions in a conll file wich contains both gold labels and machine annotations.
This file prints out the confusion matrix and the overall macro precision, recall and F1, as well as the micro F1 score.
Additionally, the precision, recall and F1 are computed per class, and a table is printed out that gives an overview of these scores.

This file can be run as follows:    
`python .\feature_extraction.py '[conll_input_file_path]' [gold_index] [predictions_index]`    
as for instance
`python evaluate.py '../data/SRL_argument_test_predictions.conll' -2 -1`
Where `'[conll_input_file_path]'` is the file  that contains the gold labels and the predictions, `[gold_index]` the index of the column in which the gold annotations can be found, and `[predictions_index]` is the index of the column that contains the machine predictions.

----------------------
## Task description and results for the different tasks

### PREDICATE EXTRACTION

#### Task description
Predicate extraction is a token binary classification tasks, that tries to predict whether each token in an utterance has the role of the predicate or not. It provides the ground for the argument recognition and classification task, since the predicates provide the stem around which the arguments are formed and acquire meaning.      

#### Implementation sequence.
To perform this task we use both a rule-based approach and machine learning method. Below we outline the steps we take for data preprocessing, feature extraction, training and testing, followed by a Results section where the different approaches are evaluated and compared.     

* We processes the original train and test/dev datasets, clean them up and extract new features using the Spacy library. In our new format, each token gets its own row, where further the following information from the gold data is represented: 
  * the sentence id
  * the token id (i.e. position in the sentence)      
Continuing, the followng features are extracted using Spacy and added to the dataset:
  * the lemma
  * the PoS label
  * the PoS tag
  * the prevous token
  * the PoS of the previous token
  * the dependency label
  * the head of the token (represented as a token)      
Finally, the last column of the processed datasets has the value 1 if the token is predicate and 0 if not.      
    
* We extract the rows of the train and test datasets into lists of lines, and prune them to account for the imbalances in the dataset (the percentage of non-predicate labels is far greater than that of predicate ones) by extracting those data points that are unlikely to be predicates, namely punctuation tokens (such as "&") and numerals (e.g. "99"). We find the numerals either with the python function `isalnum`, or by the PoS tag `CD` which represents cardinal numbers.      
* For the rule based method we print out the distribution of the labels for each feature category, to get an insight into the labels that are the most common for predicates. This will help us decide the rules for determining whether a token is a predicate or not. We notice there is a greater imbalance in the distribution of labels for the PoS and the dependency categories, so we decide to use the majority labels for those features, since it makes the distinguishing of predicates easier. In particular, we extract the tokens to be predicates that satisfy at leas one of the following conditions: They have a PoS label `VERB`/`AUX` or/and a dependency label of `ROOT`. A table with the most frequent labels for the three features (PoS, Dep, PoS tag) presented in a descending order can be found below.     

| PoS  | Dep   | PoS_tags |
|------|-------|----------|
| VERB | ROOT  | VB       |
| AUX  | conj  | VBP      |
| NOUN | aux   | VBZ      |
| ADJ  | ccomp | NN       |

* For the machine learning method we provide two options, both of which use a Support Vector Machine (SVM) to train the classifier. The first is without word-embeddings, where we only include the features extracted above, and use a linear kernel. The second uses word-embeddings to represent tokens, combined with other the sparse features(extracted above) and a gaussian kernel.     

#### Results
* For the evaluation of the rule based method a classification report is used with the recall, precision and F-score for each label, as well as their averages. It is observed that the F-score of the predicate class is 0.2 points lower than that of the non-predicate class. This makes sense, since the rules we define do not account for all the predicates in the dataset. If we included more rules, then the recall of the minor class would probably be better but the precision would drop, since the feature overlap between the two classes would be higher. The evaluation table is found below:      

|              | precision | recall    |  f1-score | support |
|--------------|-----------|-----------|:---------:|:-------:|
| 0            | 0.9345648 | 0.9527609 | 0.9435752 |   9399  |
| 1            | 0.8061135 | 0.7464618 | 0.7751417 |   2473  |
| accuracy     |           |           | 0.9097877 |  11872  |
| macro avg    | 0.8703392 | 0.8496114 | 0.8593584 |  11872  |
| weighted avg | 0.9078077 | 0.9097877 | 0.9084896 |  11872  |

* For the evaluation of the Machine Learning classifier a classification report is used, with the recall, precision and f-score for each label, as well as their averages.      
  * The recall score for the precision label is 1, while the precision one is 0.5. The SVM classifier without word embeddings + linear kernel returns 0 score for the minority label. The SVM classifier with word-embeddings and a gaussian kernel performs slightly better returning a very low score for the minority class.  When is due to the bias caused by the imbalanced distribution of the labels. The score does not improve even after the pruning of the datasets.      
* To overcome the imbalance issue and in general improve the results a series of experiments and a feature ablation is carried out. The results are described below, however, to avoid a long execution, only the best ML model is included in the final code.     
      - We change the class_weight parameter to "balanced" (default = None) during the instantiation of the classifier, to account for the imbalanced dataset. This parameter performs the following calculation: n_samples / (n_classes * np.bincount(y). However the scores remain the same for both options, only that now the majority class provides 0 scores for the linear and very low scored for the gaussian model.    
      - We try oversampling the minority class using the imblearn package and we remove the specification of the class_weight parameter. Now the two classes have the same number of datapoints. The results of the linear kernel remain the same, but those of the gaussian one improves slightly.     
      - Given the above results we decide to perform feature ablation only to the classifier using word-embeddings. We find that the best results are produced when combine word-embeddings with the PoS and dependency label features. For the non-predicate class we have a precision of 0.5. but a really low recall at 0.003. For the predicates class precision is at 0.5, but recall is extremely high at 0.9. The evaluation table for this last model (the one with the best results among ML classifier) is found below.     

      Training and evaluation of SVM classifier...

|              | precision | recall    | f1-score  | support |
|--------------|-----------|-----------|-----------|:-------:|
| 0            | 0.5686275 | 0.0030854 | 0.0061376 |   9399  |
| 1            | 0.5001867 | 0.9976593 | 0.6663114 |   9399  |
| accuracy     |           |           | 0.5003724 |  18798  |
| macro avg    | 0.5344071 | 0.5003724 | 0.3362245 |  18798  |
| weighted avg | 0.5344071 | 0.5003724 | 0.3362245 |  18798  |

In conclusion, the performance of the rule based approach is far better than the ML one. This is mainly because of the highly imbalanced dataset, but especially due to the overlapping of the features in both classes. In other words, the ML systems cannot easily form patterns in the dataset, since the features found in predicates are frequently found also in the rest of the tokens. Future implementation of the task could use more elaborate features, such as constituency labels, the dependency label of the previous and next tokens, the descendants or children of each token, to test whether the results improve.



### ARGUMENT CLASSIFICATION

#### Task description

#### Used Features and Further Feature Extraction

#### Machine Learning
To train our model, we use a Support Vector Machine (SVM). Pradhan et al. (2005) find good results in SRL prediction using this model, which was an indication for us that it might also yield good results in our case.

#### Results

#### References
Pradhan, S., Ward, W., Hacioglu, K., Martin, J. H., & Jurafsky, D. (2005).
Semantic role labelingusing different syntactic views.
In Proceedings of the 43rd annual meeting of the association for computational linguistics (acl’05)(pp. 581–588)