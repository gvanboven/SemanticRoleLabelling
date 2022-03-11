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
  * the token
  * the lemma
  * the PoS label
  * the PoS tag
  * the prevous token
  * the PoS of the previous token
  * the dependency label
  * the head of the token (represented as a token)      
Finally, the last column of the processed datasets has the value 1 if the token is predicate and 0 if not.      
    
* We extract the rows of the train and test datasets into lists of lines, and prune them to account for the imbalances in the dataset (the percentage of non-predicate labels is far greater than that of predicate ones) by extracting those data points that are unlikely to be predicates, namely punctuation tokens (such as "&") and numerals (e.g. "99"). 

* For the rule based method we must define the rules manually. In order to do this, we print out the distribution of the labels for each feature category for the tokens that are predicates, to get an insight into the labels that are the most common for predicates. This will help us decide the rules for determining whether a token is a predicate or not.       
We notice there is a greater imbalance in the distribution of labels for the PoS and the dependency categories compared to the other feature categories: for instance, these is a very high number of predicates with the PoS `VERB` and `AUX`, while the number of predicates with other PoS labels is much lower. For this reason, we decide to use the majority labels for those features, since it makes the distinguishing of predicates easier. **In particular, we extract the tokens to be predicates that satisfy at least one of the following conditions: They have a PoS label `VERB`/`AUX` or/and the dependency label `ROOT`.**       

A table with the most frequent labels (presented in a descending order) for the features PoS, Dependenpency label (Dep) and PoS tag (PoS_tags) of datapoints that represent predicates in the dataset, can be found below.     

| PoS  | Dep   | PoS_tags |
|------|-------|----------|
| VERB | ROOT  | VB       |
| AUX  | conj  | VBP      |
| NOUN | aux   | VBZ      |
| ADJ  | ccomp | NN       |

* For the machine learning method we provide two options, both of which use a Support Vector Machine (SVM) to train the classifier. The first is without word-embeddings, where we only include the features extracted above, and use a linear kernel because the dimensionality of the features is very high already. More precisely, this means that we represent the token and all the other features as one-hot encoded and concatenate the vectors (except for the sentence id and the token id, which are integers). The second uses word-embeddings to represent tokens, combined with the other sparse features (extracted above) except for the tokens and lemmas and a gaussian kernel because the dimensionality of the features is reduced here (since the word-embeddings are more dense). Because we expect the word-embeddings to capture more information than one-hot encodings, we expect the second model to outperform the first.    

#### Results
* For the evaluation of the rule based method a classification report is used with the recall, precision and F-score for each label, as well as their averages. It is observed that the F-score of the predicate class is 0.2 points lower than that of the non-predicate class. This makes sense, since the rules we define do not account for all the predicates in the dataset. If we included more rules, then the recall of the minor class would probably be better but the precision would drop, since the feature overlap between the two classes would be higher. The evaluation table is found below:      

|              | precision | recall    |  f1-score | support |
|--------------|-----------|-----------|:---------:|:-------:|
| 0            | 0.9345648 | 0.9527609 | 0.9435752 |   9399  |
| 1            | 0.8061135 | 0.7464618 | 0.7751417 |   2473  |
| accuracy     |           |           | 0.9097877 |  11872  |
| macro avg    | 0.8703392 | 0.8496114 | 0.8593584 |  11872  |
| weighted avg | 0.9078077 | 0.9097877 | 0.9084896 |  11872  |

* For the evaluation of the Machine Learning classifiers we again consider the recall, precision and f-score for each label, as well as their averages.   

  * The SVM classifier without word embeddings + linear kernel returns a 0 score for the predicate class. The SVM classifier with word-embeddings and a gaussian kernel performs slightly better, but still returning a very low score for the predicate class.  This is due to the bias caused by the imbalanced distribution of the labels. The score does not improve even after the pruning of the datasets.      
  * To overcome the imbalance issue and in general improve the results a series of experiments and a feature ablation is carried out. The results are described below, however, to avoid a long execution, only the best ML model is included in the final code.     
    * We change the `class_weight` parameter to `balanced` (default = `None`) during the instantiation of the classifier, which can be used to account for imbalanced datasets. However the scores remain the same for both classifiers, with the only difference that now the non-predicate class provides 0 scores for the linear and very low scores for the gaussian model (whereas before it was the predicate-class that yielded a very low performance).    
    * We try oversampling the minority class using the `imblearn` package and we remove the specification of the `class_weight` parameter. Now the two classes have the same number of datapoints. The results of the linear kernel remain the same, but those of the gaussian one improves slightly.     
    * Given the above results we decide to perform feature ablation only to the classifier using word-embeddings. We find that the best results are produced when we combine word-embeddings with only the PoS and dependency label features. For the non-predicate class we obtain a precision of 0.5, but a really low recall at 0.003. For the predicates class precision is at 0.5, but recall is very high, at 0.9. The evaluation table for this last model (the one with the best results among ML classifier) is found below.     

Training and evaluation of the best SVM classifier (word-embeddings + PoS + dependency label):

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
Argument classification is the task to classify and assign argument types of the give predicate(s) in the sentence based on the the slected features. In this task, machine learning approache is employed to carry out the classification. The training and evaluation of the prediction in the task use gold predicates data from the original data. For the features, this task extracts some features from the original files and also use SpaCy to extract extra features such as dependency relation and head of the dependency based on the sentences. 


#### Used Features and Further Feature Extraction
To implement the classification, features are extracted from the original data, and extra features from using SpaCy to get sentences features. 

##### Features from original data
We map through the raw data use the number of tokens to locate the predicate position and other predicate info. The features extracted from the oringinal dasta are as follow:
* token index : the position of the token in the sentence
* token : the token itself
* head_lemma: then root of the sentence
* predicate descendant: the children of the predicate according to dependency relation
* path_to_predicate_length: the number of steps from the current token to the predicate, if there is no direct path to the predicate, the token is not descendant of the predicate
* predicate_lemma: the lemma of the predicate 
* predicate_index: token index of the predicate
* predicate_pos: part of speech of the predicate
* predicate_dependency: dependency relation of the predicate


The selection reason of features from original data:
* We selected token_index, path_to_prediate_length and predicate_index to locate the position relation from the aspect of index, assuming that the model will learn the dependceny of sentence by learning the index of tokens and predicates. 
* We seleted token, predicate descendant as corelation so that the model would learn that the relation of the token and predicate from the perspective of tokens.
* We also provide predicate_pos and predicate_dependency for the model to refer from the pespective of syntax and dependency relation respectively.

#### Further Feature Extraction
We concatonate the tokens as sentence and use SpaCy to get further features and map through the sentences to lexical features. 
* head_text: token of the sentence's root
* prev_token: previous token of the current token
* pos: part of speech of the current token
* postag: pos tag of the current token
* prev_pos: previous token's part of speech
* dependency: current token's dependency relation


The selection of reasons of the further features:
* We use the head_text to coordinate with the token, since we only have the predicate_lemma but not lemma for all tokens, when doing feature ablation, we would like to discover whether head_text and token are correlated to help the model distinguish the dependency relation
* We add prev_token, prev_pos to reveal the correaltion of descendants and predicate from lexical and syntactic perspectives.
* We provide dependency for every token to reveal relation of the token and predicate from the perspective of dependency relationship. (as supplements for predicate_dependency)



#### Machine Learning
To train our model, we use a Support Vector Machine (SVM). Pradhan et al. (2005) find good results in SRL prediction using this model, which was an indication for us that it might also yield good results in our case.

#### Results
Regarding the evaluation of the model, we will concern the recall, precision and F-score for each label and macro and overall performance scores: macro precision, macro recall, macro f-score and micro f-score.

We use all features extracted (features from original data and further features). Since the memory to run train all rows in the original dataset requires more than 60GB, we would only use 100000 rows from the original data to do the emperiment and get the results. Feature ablation is also employed in the experiment to compare performance of the models with different features fed in.



All Features experiment

|Overall performance | scores| 
|--------------------|-------|
|Macro precision     | 26.01 |
|Macro recall        | 16.64 |
|Macro F1 score      | 20.29 |
|Micro F1 score      | 72.18 |

performance scores per class:
|                |precision| recall  |f-score|
|----------------|---------|---------|-------|
| _              |  0.966  |  0.751  | 0.845 |
| ARG1           |  0.273  |  0.485  | 0.349 |
| ARG2           |  0.256  | 0.388   | 0.308 |
| ARG0           | 0.482   | 0.529   | 0.504 |
| ARG4           |  0.600  | 0.167   | 0.261 |
| ARGM-DIS       |  0.567  | 0.404   | 0.472 |
| ARGM-TMP       |  0.508  | 0.343   | 0.410 |
| R-ARGM-TMP     |  0.000  | 0.000   | 0.000 |
| ARGM-MOD       |  0.802  | 0.762   | 0.781 |
| ARGM-ADV       |  0.222  | 0.281   | 0.248 |
| ARGM-LOC       |  0.393  | 0.230   | 0.290 |
| ARG3           |  0.211  | 0.058   | 0.091 |
| R-ARG0         |  0.571  | 0.345   | 0.430 |
| ARGM-NEG       |  0.696  | 0.627   | 0.660 |
| V              |  0.000  | 0.000   | 0.000 |
| ARGM-ADJ       |  0.570  | 0.480   | 0.521 |
| ARGM-GOL       |  0.333  | 0.050   | 0.087 |
| ARGM-PRP       |  0.200  | 0.104   | 0.137 |
| ARGM-PRR       |  0.312  | 0.149   | 0.202 |
| ARGM-LVB       |  0.277  | 0.269   | 0.273 |
| R-ARG1         |  0.750  | 0.188   | 0.301 |
| ARGM-PRD       |  0.500  | 0.100   | 0.167 |
| ARGM-MNR       |  0.229  | 0.180   | 0.202 |
| C-ARG1         |  0.022  | 0.128   | 0.038 |
| ARGM-CAU       |  0.286  | 0.051   | 0.087 |
| ARGM-EXT       |  0.500  | 0.202   | 0.288 |
| R-ARGM-MNR     |  0.000  | 0.000   | 0.000 |
| R-ARGM-LOC     |  0.000  | 0.000   | 0.000 |
| C-V            |  0.167  | 0.077   | 0.105 |
| ARGM-COM       |  0.000  | 0.000   | 0.000 |
| ARGM-CXN       |  1.000  | 0.091   | 0.167 |
| C-ARGM-CXN     |  0.000  | 0.000   | 0.000 |
| R-ARGM-ADJ     |  0.000  | 0.000   | 0.000 |
| ARGM-DIR       |  0.010  | 0.048   | 0.017 |
| C-ARG3         |  0.000  | 0.000   | 0.000 |
| ARG5           |  0.000  | 0.000   | 0.000 |
| R-ARG2         |  0.000  | 0.000   | 0.000 |
| R-ARGM-ADV     |  0.000  | 0.000   | 0.000 |
| R-ARGM-DIR     |  0.000  | 0.000   | 0.000 |
| C-ARG0         |  0.000  | 0.000   | 0.000 |
| C-ARG2         |  0.000  | 0.000   | 0.000 |
| ARG1-DSP       |  0.000  | 0.000   | 0.000 |
| C-ARG1-DSP     |  0.000  | 0.000   | 0.000 |
| C-ARGM-LOC     |  0.000  | 0.000   | 0.000 |
| ARGA           |  0.000  | 0.000   | 0.000 |

* The performance scores of the all features fed in model is poor and we may need to do some feature ablation experiments to see which feaures are useful for the model to distinguish the dependency relation. Besides, the representations of the features do not strongly reveal the dependency relations between tokens and predicates. Since there are a lots of labels for the model to classify, we will focus on the main arguments of the sentences: ARG0, ARG1, ARG2, ARG4 as they are common in predicate arguments. ARG1 is the most commonly use in predicates arguments classification. In this experiment, ARG0's f-score is 0.504, which is acceptabel but still needs improvement.

Only Predicate Info Features Experiment:

|Overall performance | scores|
|--------------------|-------|
|Macro precision     | 28.57 |
|Macro recall        | 15.93 |
|Macro F1 score      | 20.46 |
|Micro F1 score      | 91.69 |

performance scores per class:
|               | precision | recall  |f-score|
|---------------|-----------|----------|------|
|_              | 0.953     | 0.973    |0.963 |
|ARG1           | 0.570     | 0.395    |0.467 |
|ARG2           | 0.301     | 0.215    |0.251 |
|ARG0           | 0.558     | 0.515    |0.536 |
|ARG4           | 0.600     | 0.167    |0.261 |
|ARGM-DIS       | 0.710     | 0.394    |0.507 |
|ARGM-TMP       | 0.455     | 0.432    |0.443 |
|R-ARGM-TMP     | 0.000     | 0.000    |0.000 |
|ARGM-MOD       | 0.822     | 0.681    |0.745 |
|ARGM-ADV       | 0.267     | 0.261    |0.264 |
|ARGM-LOC       | 0.298     | 0.250    |0.272 |
|ARG3           | 0.132     | 0.070    |0.091 |
|R-ARG0         | 0.657     | 0.371    |0.474 |
|ARGM-NEG       | 0.771     | 0.570    |0.655 |
|V              | 0.000     | 0.000    |0.000 |
|ARGM-ADJ       | 0.238     | 0.479    |0.318 |
|ARGM-GOL       | 0.500     | 0.130    |0.206 |
|ARGM-PRP       | 0.304     | 0.099    |0.149 |
|ARGM-PRR       | 0.647     | 0.162    |0.259 |
|ARGM-LVB       | 0.000     | 0.000    |0.000 |
|R-ARG1         | 0.750     | 0.143    |0.240 |
|ARGM-PRD       | 0.667     | 0.095    |0.166 |
|ARGM-MNR       | 0.242     | 0.165    |0.196 |
|C-ARG1         | 0.304     | 0.146    |0.197 |
|ARGM-CAU       | 0.250     | 0.048    |0.081 |
|ARGM-EXT       | 0.500     | 0.208    |0.294 |
|R-ARGM-MNR     | 0.000     | 0.000    |0.000 |
|R-ARGM-LOC     | 0.000     | 0.000    |0.000 |
|C-V            | 0.250     | 0.071    |0.111 |
|ARGM-COM       | 0.000     | 0.000    |0.000 |
|ARGM-CXN       | 1.000     | 0.083    |0.153 |
|C-ARGM-CXN     | 0.000     | 0.000    |0.000 |
|R-ARGM-ADJ     | 0.000     | 0.000    |0.000 |
|ARGM-DIR       | 0.111     | 0.047    |0.066 |
|C-ARG3         | 0.000     | 0.000    |0.000 |
|ARG5           | 0.000     | 0.000    |0.000 |
|R-ARG2         | 0.000     | 0.000    |0.000 |
|R-ARGM-ADV     | 0.000     | 0.000    |0.000 |
|R-ARGM-DIR     | 0.000     | 0.000    |0.000 |
|C-ARG0         | 0.000     | 0.000    |0.000 |
|C-ARG2         | 0.000     | 0.000    |0.000 |
|ARG1-DSP       | 0.000     | 0.000    |0.000 |
|C-ARG1-DSP     | 0.000     | 0.000    |0.000 |
|C-ARGM-LOC     | 0.000     | 0.000    |0.000 |
|ARGA           | 0.000     | 0.000    |0.000 |

* We would like to see how does the model perform with only predicate info features from original dataset fed in the model because we would like to see wether the further features are useful for the model to distinguish the token and predicate relation. From the overall performance, we can see that the macro f1-score is slightly higher (0.17) than the performance of all features fed model. Besides, the f-score of ARG0 is higher(0.032) than all features fed in model. We assume that some features are noisy for the classification.

Only further features experiments:

|Overall performance | scores|
|--------------------|-------|
|Macro precision     | 25.01 |
|Macro recall        | 16.4  |
|Macro F1 score      | 19.81 |
|Micro F1 score      | 89.75 |

performance scores per class:
|               | precision | recall  |f-score|
|---------------|-----------|----------|------|
|_              | 0.957     |0.947     |0.952 |
|ARG1           | 0.388     |0.421     |0.404 |
|ARG2           | 0.286     |0.164     |0.208 |
|ARG0           | 0.396     |0.569     |0.467 |
|ARG4           | 0.600     |0.170     |0.265 |
|ARGM-DIS       | 0.676     |0.399     |0.502 |
|ARGM-TMP       | 0.258     |0.445     |0.327 |
|R-ARGM-TMP     | 0.000     |0.000     |0.000 |
|ARGM-MOD       | 0.751     |0.730     |0.740 |
|ARGM-ADV       | 0.236     |0.288     |0.259 |
|ARGM-LOC       | 0.345     |0.266     |0.300 |
|ARG3           | 0.174     |0.062     |0.091 |
|R-ARG0         | 0.679     |0.328     |0.442 |
|ARGM-NEG       | 0.656     |0.588     |0.620 |
|V              | 0.000     |0.000     |0.000 |
|ARGM-ADJ       | 0.287     |0.500     |0.365 |
|ARGM-GOL       | 0.500     |0.095     |0.160 |
|ARGM-PRP       | 0.261     |0.090     |0.134 |
|ARGM-PRR       | 0.579     |0.169     |0.262 |
|ARGM-LVB       | 0.000     |0.000     |0.000 |
|R-ARG1         | 0.750     |0.150     |0.250 |
|ARGM-PRD       | 0.625     |0.135     |0.222 |
|ARGM-MNR       | 0.242     |0.200     |0.219 |
|C-ARG1         | 0.269     |0.159     |0.200 |
|ARGM-CAU       | 0.027     |0.079     |0.040 |
|ARGM-EXT       | 0.488     |0.208     |0.292 |
|R-ARGM-MNR     | 0.000     |0.000     |0.000 |
|R-ARGM-LOC     | 0.000     |0.000     |0.000 |
|C-V            | 0.200     |0.077     |0.111 |
|ARGM-COM       | 0.000     |0.000     |0.000 |
|ARGM-CXN       | 0.500     |0.091     |0.154 |
|C-ARGM-CXN     | 0.000     |0.000     |0.000 |
|R-ARGM-ADJ     | 0.000     |0.000     |0.000 |
|ARGM-DIR       | 0.125     |0.049     |0.070 |
|C-ARG3         | 0.000     |0.000     |0.000 |
|ARG5           | 0.000     |0.000     |0.000 |
|R-ARG2         | 0.000     |0.000     |0.000 |
|R-ARGM-ADV     | 0.000     |0.000     |0.000 |
|R-ARGM-DIR     | 0.000     |0.000     |0.000 |
|C-ARG0         | 0.000     |0.000     |0.000 |
|C-ARG2         | 0.000     |0.000     |0.000 |
|ARG1-DSP       | 0.000     |0.000     |0.000 |
|C-ARG1-DSP     | 0.000     |0.000     |0.000 |
|C-ARGM-LOC     | 0.000     |0.000     |0.000 |
|ARGA           | 0.000     |0.000     |0.000 |

* We also train a model with only further features fed in and the macro f-score and ARG0 f-score is lower than the only predicate info fed in model, which means that there are some noisy features in further features and if we want to improve the performance, we need to select certain features from further features set to combine with the predicate info features.

Selected certain features: 
|features                 |data origin|
|-------------------------|---------|      
|token                    | original|           
|head_lemma               | original|              
|predicate_descendant     | original|                
|path_to_predicate_length | original|                 
|predicate_index          | original|               
|predicate_lemma          | original|                   
|predicate_pos            | original|                    
|predicate_dependency     | original|                        
|prev_pos                 | further |                 
|pos                      | further |   

|Overall performance | scores|
|--------------------|-------|
|Macro precision     | 28.88 |
|Macro recall        | 15.91 |
|Macro F1 score      | 20.51 |
|Micro F1 score      | 90.66 |

performance scores per class:
|               | precision | recall  |f-score|
|---------------|-----------|----------|------|
|_              | 0.956     | 0.959    | 0.957|
|ARG1           | 0.586     | 0.385    | 0.465|
|ARG2           | 0.246     | 0.352    | 0.290|
|ARG0           | 0.473     | 0.533    | 0.501|
|ARG4           | 0.615     | 0.148    | 0.239|
|ARGM-DIS       | 0.676     | 0.406    | 0.507|
|ARGM-TMP       | 0.436     | 0.450    | 0.443|
|R-ARGM-TMP     | 0.000     | 0.000    | 0.000|
|ARGM-MOD       | 0.802     | 0.696    | 0.745|
|ARGM-ADV       | 0.260     | 0.253    | 0.256|
|ARGM-LOC       | 0.462     | 0.228    | 0.305|
|ARG3           | 0.250     | 0.070    | 0.109|
|R-ARG0         | 0.700     | 0.344    | 0.461|
|ARGM-NEG       | 0.774     | 0.551    | 0.644|
|V              | 0.000     | 0.000    | 0.000|
|ARGM-ADJ       | 0.146     | 0.486    | 0.225|
|ARGM-GOL       | 0.667     | 0.087    | 0.154|
|ARGM-PRP       | 0.273     | 0.086    | 0.131|
|ARGM-PRR       | 0.625     | 0.147    | 0.238|
|ARGM-LVB       | 0.000     | 0.000    | 0.000|
|R-ARG1         | 0.750     | 0.146    | 0.244|
|ARGM-PRD       | 0.667     | 0.098    | 0.171|
|ARGM-MNR       | 0.275     | 0.169    | 0.209|
|C-ARG1         | 0.263     | 0.106    | 0.151|
|ARGM-CAU       | 0.286     | 0.050    | 0.085|
|ARGM-EXT       | 0.513     | 0.200    | 0.288|
|R-ARGM-MNR     | 0.000     | 0.000    | 0.000|
|R-ARGM-LOC     | 0.000     | 0.000    | 0.000|
|C-V            | 0.200     | 0.077    | 0.111|
|ARGM-COM       | 0.000     | 0.000    | 0.000|
|ARGM-CXN       | 1.000     | 0.083    | 0.153|
|C-ARGM-CXN     | 0.000     | 0.000    | 0.000|
|R-ARGM-ADJ     | 0.000     | 0.000    | 0.000|
|ARGM-DIR       | 0.095     | 0.048    | 0.064|
|C-ARG3         | 0.000     | 0.000    | 0.000|
|ARG5           | 0.000     | 0.000    | 0.000|
|R-ARG2         | 0.000     | 0.000    | 0.000|
|R-ARGM-ADV     | 0.000     | 0.000    | 0.000|
|R-ARGM-DIR     | 0.000     | 0.000    | 0.000|
|C-ARG0         | 0.000     | 0.000    | 0.000|
|C-ARG2         | 0.000     | 0.000    | 0.000|
|ARG1-DSP       | 0.000     | 0.000    | 0.000|
|C-ARG1-DSP     | 0.000     | 0.000    | 0.000|
|C-ARGM-LOC     | 0.000     | 0.000    | 0.000|
|ARGA           | 0.000     | 0.000    | 0.000|

* We combine the features from original data and further features to train the model. The macro f-score (20.51) is the highest among all. Even though the f-score of ARG0 is not the highest among all, but the over all performance of ARG0, ARG1, ARG4 is acceptable.

To conclude, the overall performance of all the models above is not satisfying as the macro recall, macro precision, macro f-score are not as high as the other classifiers which can achieve more than 50 in macro scores. The micro scores of the models are higher than 70 because the label'_' appears a lot in the dataset, which means the dataset is biased.
We need to look for the benchmark of the srl task using SVM to compare the performance of our models.

























#### References
Pradhan, S., Ward, W., Hacioglu, K., Martin, J. H., & Jurafsky, D. (2005).
Semantic role labelingusing different syntactic views.
In Proceedings of the 43rd annual meeting of the association for computational linguistics (acl’05)(pp. 581–588)
