
# Semantic Role Labelling

In this repository, our code can be found to implement a Semantic Role Labelling System.   
We have created this code for the second assignment for the course NLP Technologies at the Vrije Universiteit Amsterdam.  
We train our system on data from the Universal Proposition Banks dataset.   
This readme consists of two parts. In the first part we present a description of the code and how to run it. In the second part we describe our task and present our results.  
This project is carries out by Alice Ye, Vicky Kyrmanidi and G. van Boven.

----------------------
## Repository description

This repository consists of 3 folders, which we will each describe.

#### `.\data`
This folder contains the train, dev and test data we use for our model.

#### `.\models`
This folder contains the W2V word embeddings model we use to encode tokens during training. This model is currently excluded because of its large size.    
The model should thus be downloaded and placed in this folder in order for the code to work. The model can be downloaded from https://code.google.com/archive/p/word2vec/

#### `.\code`

This folder contains our code. In this fodler you find a separate README in which we describe all of our files. 

----------------------
## Task description and results for the different tasks

In this project, we implement a system to carry out the task of Semantic Role Labelling (SRL). Theoretically, the goal of this task is to answer the question 'who did what to whom', in other words to find the participants of an event, each of which has a specific role. For an automated SRL system, this means that a model should return the predicates and its argument, for a given input sentence. Here, the arguments are specific roles that participants of an event can take: e.g. agents, patient, instrument, etc. For instance, in the sentence "Vicky helped Goya", "Vicky" is the agent, "Goya" is the patient, and "help" is the predicate.

#### Data description
In this project, we train an SRL system on data from the Universal Proposition Banks, which is an annotated corpus which uses its own labels for defining the semantic roles, following the Propbank annotation scheme. In this scheme `ARG0` represents the agent, `ARG1` represents the patient, `ARG2` indicates the instrument, `ARGM-MNR` for the manner, `ARGM-LOC` for the location and `ARGM-TMP` for the time.     
From this dataset, we take a train, dev and test dataset, which are all in English. The annotations in this dataset are a mix of human annotations, machine annotations and machine annotations checked by humans. Even though these annotations cannot technically be considered to be gold, we do refer to them as gold in the rest of our report. 

#### Task description
We can consider the task to consist of two steps: (i) finding the predicates present in a given sentence, and (ii) extracting and classifying the corresponding arguments for each predicate. 
In this project, we split up these two tasks, and evaluate the outputs separately with the gold data. More precisely, in taks (i) we implement a rule-based and a machine-learning based predicate extraction method, and compare their outputs to the predicates in the gold data. For this task we must keep in mind that the data is very skewed: the dataset consists of much more non-predicate tokens than predicates.      
Continuing, we approach task (ii) with a machine learning method using a Support Vector Machine. Here, we use all the tokens as input, together with the respective predicate and additional features that we extract, and aim to predict what argument label belongs to the token. We evaluate our results again against the gold data. This thus means that we do **not** use our outputs for task (i) to train task (ii), rather we include the gold-predicates in task (ii). This means that our scores for task (ii) overestimates the actual scores, as in reality here, our scores for task (ii) can only be as good as our best score for task (i). Again for this task, it is important that we keep in mind that the data is skewed towards non-argument tokens. Finally, for the second task we also implement an LSTM-based neural network using AllenNLP, which we describe in part 3 of this README below. 

For both task we had to preprocess the data. The steps we took in order to do this are described in the corresponding sections.

----------------------
## 1. PREDICATE EXTRACTION

#### Task description
Predicate extraction is a token binary classification tasks, that tries to predict whether each token in an utterance has the role of the predicate or not. It provides the ground for the argument recognition and classification task, since the predicates provide the stem around which the arguments are formed and acquire meaning.      

To perform this task we use both a rule-based approach and machine learning method. Below we outline the steps we take for data preprocessing, feature extraction, training and testing, followed by a Results section where the different approaches are evaluated and compared.     

#### Implementation sequence.
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

#### Rule-based approach
For the rule based method we must define the rules manually. In order to do this, we print out the distribution of the labels for each feature category for the tokens that are predicates, to get an insight into the labels that are the most common for predicates. This will help us decide the rules for determining whether a token is a predicate or not.       
We notice there is a greater imbalance in the distribution of labels for the PoS and the dependency categories compared to the other feature categories: for instance, these is a very high number of predicates with the PoS `VERB` and `AUX`, while the number of predicates with other PoS labels is much lower. For this reason, we decide to use the majority labels for those features, since it makes the distinguishing of predicates easier. **In particular, we extract the tokens to be predicates that satisfy at least one of the following conditions: They have a PoS label `VERB`/`AUX` or/and the dependency label `ROOT`.**       

A table with the most frequent labels (presented in a descending order) for the features PoS, Dependenpency label (Dep) and PoS tag (PoS_tags) of datapoints that represent predicates in the dataset, can be found below.     

| PoS  | Dep   | PoS_tags |
|------|-------|----------|
| VERB | ROOT  | VB       |
| AUX  | conj  | VBP      |
| NOUN | aux   | VBZ      |
| ADJ  | ccomp | NN       |

#### Machine learning approach
For the machine learning method we provide two options, both of which use a Support Vector Machine (SVM) to train the classifier. The first is without word-embeddings, where we only include the features extracted above, and use a linear kernel because the dimensionality of the features is very high already. More precisely, this means that we represent the token and all the other features as one-hot encoded and concatenate the vectors (except for the sentence id and the token id, which are integers). The second uses word-embeddings to represent tokens, combined with the other sparse features (extracted above) except for the tokens and lemmas and a gaussian kernel because the dimensionality of the features is reduced here (since the word-embeddings are more dense). Because we expect the word-embeddings to capture more information than one-hot encodings, we expect the second model to outperform the first.    

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


#### Conclusion
In conclusion, the performance of the rule based approach is far better than the ML one. This is mainly because of the highly imbalanced dataset, but especially due to the overlapping of the features in both classes. In other words, the ML systems cannot easily form patterns in the dataset, since the features found in predicates are frequently found also in the rest of the tokens. Future implementation of the task could use more elaborate features, such as constituency labels, the dependency label of the previous and next tokens, the descendants or children of each token, to test whether the results improve.


----------------------
## 2. ARGUMENT CLASSIFICATION

#### Task description
Argument classification is the task to classify and assign argument types of a given predicate(s) in a sentence based on the the slected features. In this task, we employ a machine learning approach to carry out the classification task. The training and evaluation of the prediction in the task use gold predicates data from the original data. For the features, this task extracts some features from the gold data file and also use SpaCy to extract extra features such as dependency relation and head of the dependency based on the sentences. 


#### Used Features and Further Feature Extraction
In order to use this dataset, we must first preprocess it. In the original datasets, tokens have their own rows, sentences are separated by empty lines, and predicates and their arguments are presented in columns. In the restructured version of the data, the empty lines are taken out, and each token-predicate combination gets its own row. This means that if a sentence contains 10 tokens and 3 predicates, this sentence will be represented over 10 * 3 = 30 rows. If a sentence contains no predicates, it is taken out. 
To implement the classification, features are extracted from the original (gold) data, and extra features are extracted by using SpaCy to get sentences features. 

##### Features from original data
We map through the raw data use the number of tokens to locate the predicate position and other predicate info. The features extracted from the oringinal dasta are the following:
| Name                     | Description                                                                                                                                                                                                              | Data type        |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| token_index              | the position of the token in the sentence (integer)                                                                                                                                                                      |      integer     |
| token                    | the token itself (string)                                                                                                                                                                                                |      string      |
| head_lemma               | the lemma of the head of the token (string)                                                                                                                                                                              |      string      |
| predicate_descendant     | a binary feature, indicating whether the current token is a descendant of the  current predicate (1), or not (0) (integer)                                                                                               | integer (binary) |
| path_to_predicate_length | the number of steps from the current token to the predicate in the dependency tree,  if there is no direct path to the predicate, the token is not descendant of the predicate  and then the value will be 100 (integer) |      integer     |
| predicate_lemma          | the lemma of the current predicate                                                                                                                                                                                       |      string      |
| predicate_index          | token index of the current predicate                                                                                                                                                                                     |      integer     |
| predicate_pos            | part of speech of the current predicate                                                                                                                                                                                  |      string      |
| predicate_dependency     | dependency label of the current predicate                                                                                                                                                                                |      string      |

For the token, the predicate lemma and the 

The selection reason of features from original data:
* We selected `token_index`, `path_to_prediate_length` and `predicate_index` because these all provide some information about the location of the token and the predicate in the sentence. We expect that it is often the case that the token and the predicate are close together in the sentence, or that the length of the dependency path between them is small.
* We selected `token`, and the `precidate_lemma` to include information on the token/lemma of the current token and the predicate.
* We include `predicate_descendant` to the model access to whether the current token is a descendant of the predicate, because this would make it more likely that the token is an argument for the predicate.
* We add the `head_lemma` to include some information about what the head is of the current token. 
* We also provide `predicate_pos` and `predicate_dependency` for the model to refer from the pespective of syntax and dependency of the predicate respectively.

#### Further Feature Extraction
We concatenate the tokens as sentence and use SpaCy to get further features and map through the sentences to lexical features. 
| Name       | Description                                                     | Data type |
|------------|-----------------------------------------------------------------|-----------|
| head_text  | token of the head of the current token                          |   string  |
| prev_token | the token that preceeds the current token                       |   string  |
| pos        | the Part of Speech of the current token                         |   string  |
| postag     | the Part of Speech tag of the current token                     |   string  |
| prev_pos   | the Part of Speech of the token that preceeds the current token |   string  |
| dependency | the dependency label of the current token                       |   string  |


The selection of reasons of the further features:
* We add `prev_token` and `prev_pos` include some information about the token that preceeds the current token in the sentence, which might might provide information about what argument we are dealing with in some cases (for instance, being preceded by 'with' might indicate we are dealing with an instrument)
* We add information about the Part of Speech through the `pos` and the `postag`. We think this can help the model, for instance because a noun is more likely to be an argument than a verb.
* Finally we provide the dependency label for every token, since we believe this can carry a lot of information about what argument we are dealing with.

#### Machine Learning
To train our model, we use a Support Vector Machine (SVM). Pradhan et al. (2005) find good results in SRL prediction using this model, which was an indication for us that it might also yield good results in our case. The specific task here is a multi-class classification task, where there are 45 classes to distringuish between: the `_` label,  that indicates that the current token is not an argument, and 44 argument classes. Notably, the dataset is very skewed in this respect: the `_` is very highly overrepresented, while all the other labels are much less common.      
In our model, we represent our features that are integeres as integers, and represent all the features that are string through one-hot encodings, except for the `token`, `head_lemma`, `prev_token` and the `head_text`, as we use Word2Vec word-embeddings to represent those.

#### Results
Regarding the evaluation of the model, we will consider the recall, precision and F-score for each label. Additionally, we inspect the following overall performance scores: macro precision, macro recall, macro f-score and micro f-score.

We use all features extracted (features from original data and further features). Since the memory to run train all rows in the original dataset requires more than 60GB, and this is too much for our computers, we decided to only use 100.000 rows from the data to train and test our models on. After training a model on all features, we carry out a feature ablation study (see below) to compare performance of models with different features fed in.


##### All Features experiment

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

* The performance scores of the model with all features fed is poor and we may need to do some feature ablation experiments to see which feaures are useful for the model. We expect that the information about the predicate is important, since the argument relation is established between the token and the predicate.Contuing, we expect the dependency label to be important, as this is one of the main features a human would consider when assignming semantic role labels. However, we cannot know in advance whether the model will consider the same features as humans do.       
 Since there are a lot of labels for the model to classify, we will focus on the main arguments of the sentences: `ARG0`, `ARG1`, `ARG2`, `ARG4` as they are common predicate arguments. `ARG0` is the most commonly used in predicates arguments classification. In this experiment, `ARG0`'s f-score is 0.504, which needs improvement, but it comes closer to being acceptable.

##### Only Predicate Info Features Experiment:    
Here we used only the features from the original data (so not the features we extracted from SpaCy), meaning the following: `token_index`, `token` `head_lemma`, `predicate_descendant`, `path_to_predicate_length`, `predicate_lemma`, `predicate_index`, `predicate_pos`, `predicate_dependency`.  Here, the dependency label of the token is not included. As we expect the dependency label to be important for argument classification, we expect the performance of the model to drop here.  

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

* We would like to see how does the model perform with only predicate info features from original dataset fed in the model, because we would like to see wether the further features are useful for the model to distinguish the token and predicate relation. From the overall performance, we can see that the macro f1-score is slightly higher (+0.17) than the performance of all features fed model. Besides, the f-score of `ARG0` is higher(+0.032) than the model with all the features fed. While we expected the dependencly label to be important, it appears that it is not crucial for the task (although this was not the only feature we excluded). Importantly however, the results of this model are still poor. Therefore, this selection of features does not appear to be sufficiently imformative for the model to learn this task.

##### Only further features experiments:

In this experiment we include the following features: `token`, `token index`, `head_text`, `prev_token`, `postag`, `postag`, `prev_pos`, `dependency`. Importantly, we do not include any information about the predicate in this experiment, so we expect the performance to drop further.

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

* We also train a model with only further features fed in. Here we note that the the macro f-score (-0.65) and `ARG0` f-score is lower (-0.069) than for the only predicate info model. This might indicate that the information about the predicates is useful for the model to make predictions. However, as we use a nearly completly different list of features here, we cannot be certain that it is the lack of predicate information that causes the drop in performance. Again it is important to note that the difference in performance is not so big, and the performances of the models remain in the same range. Finally, we will try to elect certain features from further features set to combine with the predicate info features, to see if this can reach a better model.

##### Selected certain features: 
In this experiment we include the following features:
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

* We combine the features from original data and further features to train the model. The macro f-score (20.51) is the highest among all. Even though the f-score of `ARG0` is not the highest among all, the overall performance of `ARG0`, `ARG1`, `ARG4` is acceptable. We consider this model to be our best model, but still its results are far from satisfactory to use in a downstream task.

Finally we also experimented with compressing our data using Singular Value Decomposition, but this did not improve our results. 

To conclude, the overall performance of all the models above is not satisfying as the macro recall, macro precision, macro f-score are not as high as we would want from our model, as we would hope it would be able to achieve a score of at least 50 in macro scores. We do find micro F1 scores higher than 70, but this is due to the fact that the label `_` is overrepresented in the dataset, and the scores for this label are much higher.       
We must also keep in mind here that these results are still an overestimation of the actual performance scores, since we use the gold predicates as inputs, rather than our extracted predicates. In reality, the performance is thus even still lower.      
So we conclude that the task as it currently is is too difficult for our model to learn. This is likely to be caused by the fact that our dataset is very unbalanced, and because our models is not able to find patterns in the data.


#### Discussion
Overall, it appears that our chosen approach might not be ideal for this task. We decided to combine the argument extraction and classification tasks into one step. But as the empty label `_` is highly overrepresented in the data, and all the other labels are much more uncommon, this task might be too difficult for our model to learn. Potentially, a better approach would have been to extract the agruments using a rule-based approach, e.g. by taking those tokens that are direct dependents of the predicate. Continuing we could have trained a machine learning model on only those token that we have extracted to be arguments, to predict the argument label. In this case, the distribution of labels would be less skewed (as there is no `_` label in the data), which might make the classification task less difficult. 

----------------------
## 3. LSTM-based Neural Network with AllenNLP






----------------------
#### References
Pradhan, S., Ward, W., Hacioglu, K., Martin, J. H., & Jurafsky, D. (2005).
Semantic role labelingusing different syntactic views.
In Proceedings of the 43rd annual meeting of the association for computational linguistics (acl’05)(pp. 581–588)
