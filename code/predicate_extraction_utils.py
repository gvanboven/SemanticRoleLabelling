from collections import Counter
import csv
import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np
import string
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state = 42)
from sklearn.metrics import accuracy_score


#the following function is taken from group 7 Konstantina Andronikou, Lahorka Nikolovski, Mira Reisinger (Text Mining) on 2022/03/07
def get_features(string):
    '''
    Function that extracts features from a string and returns a list of lists, where each list contains the features extracted for token of the sentence.

    :param string: a sentence
    :type string: string

    :returns sent_features: lists of features extracted from this sentence
    :type sent_features: list
    '''

    sent_features = []
    doc = nlp(string)
    for sent in doc.sents: # code from group 7 Konstantina Andronikou, Lahorka Nikolovski, Mira Reisinger
        for token_index, token in enumerate(sent): # adapted code from group 7 Konstantina Andronikou, Lahorka Nikolovski, Mira Reisinger
            if token_index >0 :
                pre_token = sent[token_index-1]
                pre_pos = pre_token.pos_
            else:
                pre_token = "'nan'"
                pre_pos = "'nan'"

            extra_features = [str(token.lemma_),
                              str(token.pos_),
                              str(token.tag_),
                              str(pre_token),
                              str(pre_pos),
                              str(token.dep_),
                              str(token.head.text)]


            sent_features.append(extra_features)

    return sent_features


def fix_rows(sent_rows, sent_index):
    '''
    Function that takes a list of rows, and adjusts each row to contain more features.

    :param sent_rows: a list of rows for a sentence, where each row represents a token of that sentence and its features
    :type sent_rows: a list (of lists)
    :param sent_index: the index of the current sent_rows
    :type sent_index: string

    :returns fixed_rows: the adjusted rows
    :type fixed_rows: a list (of lists)
    '''

    #We join the token of each list into a string
    sentence_string = " ".join([row[1] for row in sent_rows])

    #Extract features from the tokens in that string
    extracted_features = get_features(sentence_string)

    #We account for cases where there is no predicate in the data
    try:
        if sent_rows[0][11] == '':  #eg 'via Microsoft Watch from Mary Jo Foley'
            return False
        n_predicates = int(len(sent_rows[0]) - 11)
    except:
        #skip empty instances: the length is smaller than eleven
        return False

    fixed_rows = []
    for row, features_list in zip(sent_rows, extracted_features):
        row_list = []
        if row[10] != "_":
            predicate = '1'
        else:
            predicate = '0'

        row_list = [sent_index, row[0], row[1]]
        row_list.extend(features_list)
        row_list.append(predicate)
        fixed_rows.append(row_list)

    return fixed_rows



def new_dataset(input_file, output_file):
    '''
    Function that reads a file and modifies the format of its dataset in terms of rows and columns in order to serve a predicate extraction/prediction task. It stores the new modified dataset in a new file.

    :param input_file: the path to the input file
    :type input_file: string
    :param output_file: the path to the output file
    :type output_file: sting
    '''
    with open(input_file, 'r', encoding='utf8') as infile:
        rows = infile.readlines()

    sent_index = 0
    all_new_rows = []
    sent_rows = []

    for row in rows:
        if row.startswith('#'):
            continue

        if row == '\n':
            new_rows =  fix_rows(sent_rows, sent_index)

            if new_rows == False:
                sent_rows = []
                continue

            all_new_rows.extend(new_rows)

            sent_index += 1
            sent_rows = []
            continue

        #split row into a list
        datapoint = row.strip('\n').split('\t')
        #add current row to our sentence rows
        sent_rows.append(datapoint)

    #write the new rows into a new file
    with open(output_file, 'w', encoding="utf8") as outfile:
        writer = csv.writer(outfile, delimiter = '\t', lineterminator='\n')
        headers = ['sent_id', 'token_id', 'token', 'lemma', 'pos', 'pos_tag', 'prev_token', 'prev_pos', 'dep', 'head', 'label' ]

        writer.writerow(headers)
        writer.writerows(all_new_rows)

def pruning(lines):
    '''
    Function that takes out data points that are very inlikely to be predicates.

    :param lines: the rows of the dataset
    :type lines: a list of lists_file

    :returns lines: the pruned rows of the datasets
    :type lines: a list of lists
    '''
    for line in lines:
        conditions = [line[2] in string.punctuation, line[2].isalnum(), line[5]=='CD']
        if any(conditions):
            lines.remove(line)

    return(lines)

#This function is take from the course Machine Learning for NLP on 2022/03/07
def read_in_conll_file(conll_file: str, delimiter: str = '\t'):
    '''
    Read in conll file and return structured object

    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll

    :returns List of splitted rows included in conll file
    '''
    with open(conll_file, 'r', encoding='utf8') as infile:
        conll_data = infile.readlines()
    rows = []
    for row in conll_data:
        if row != []:
            rows.append(row.replace('\n','').split('\t'))

    #We prune the dataset
    pruned_rows = pruning(rows[1:])
    return pruned_rows

def extract_predicate_lines(lines):
    '''
    Function that takes a list of lines and extract those lines that represent predicates.

    :param lines: the rows of the dataset where each row represents a token and its features
    :type lines: a list of lists

    :returns predicate_lines: the rows that represent the predicates of the dataset along with their features
    :type predicate_lines: a list of lists
    '''
    predicate_lines = []
    for line in lines:
        if line[-1] == str(1):
            predicate_lines.append(line)

    return predicate_lines

#THE FOLLOWING FUNCTION CAN BE USED FOR EXTRACTING THE PREDICATES IN A RULE BASED METHOD
def get_distribution(lines):
    '''
    Function that takes a list of lines(lists) where each line represents a token and its features, extracts those lines that represent predicates and their features, and prints the distribution of predicate features

    :param lines: the rows of the dataset where each row represents a token and its features
    :type lines: a list of lists
    '''
    pred_lines = extract_predicate_lines(lines)

    #POS DISTRIBUTION
    #predicates
    pred_pos = [line[4] for line in pred_lines]
    predicate_pos_stats = Counter(pred_pos)
    print('Predicate PoS distribution:')
    print(predicate_pos_stats)
    #rest of data
    #data_pos = [line[4] for line in lines]
    #data_pos_stats = Counter(data_pos)
    #print('Rest of data PoS distribution:')
    #print(data_pos_stats)
    print('-'*100)
    print()

    #POS TAG DISTRIBUTION
    #predicates
    pred_postags = [line[5] for line in pred_lines]
    predicate_postags_stats = Counter(pred_postags)
    print('Predicate PoS tags distribution:')
    print(predicate_postags_stats)
    #rest of data
    #data_postags = [line[5] for line in lines]
    #data_postags_stats = Counter(data_postags)
    #print('Rest of data PoS tag distribution')
    #print(data_postags_stats)
    print('-' * 100)
    print()

    #DEPENDENCY LABEL DISTRIBUYTION
    #predicates
    #dependency labels
    pred_dep = [line[8] for line in pred_lines]
    pred_dep_stats = Counter(pred_dep)
    print('Predicate dependency label distribution')
    print(pred_dep_stats)
    #rest of data
    #data_dep = [line[8] for line in lines]
    #data_dep_stats = Counter(data_dep)
    #print('Rest of data dependency label distribution')
    #print(data_dep_stats)
    #print("-", 100)

    return pred_lines

def rule_based_pred_extraction(lines):
    predictions = []
    for line in lines:
        conditions = [line[4]=='VERB', line[4]=='AUX',line[8]=='ROOT']
        if any(conditions):
            predictions.append('1')
        else:
            predictions.append('0')

    gold_labels = [line[-1] for line in lines]
    evaluation = accuracy_score(gold_labels, predictions)
    print(evaluation)


#THE FOLLOWING FUNCTIONS ARE FOR PREDICATE PREDICTIONS USING MACHINE LEARNING

feature_to_index = {'sent_id':0, 'token_id':1, 'token':2, 'lemma':3, 'pos':4, 'pos_tag':5, 'prev_token':6, 'prev_pos':7,
                   'dep':8, 'head':9, 'label':10}

#This function is taken and adjusted from the course Machine Learning for NLP on 2022/03/07
def extract_features_and_gold_labels(lines, selected_features):
    '''
    Extract specific features from a dataset and return a list of dictionaries, where each dictionary carries the features of one datapoint. It also applies \
    oversamping to the minority class datapoints and their labels

    :param lines: the rows of a dataset
    :type lines: a list of lists
    :param selected_features: the selected features to be extracted from the lines
    :type selected_features: a list of strings

    :returns features: the features extracted from the lines
    :type features: a list of dictionaries where the keys are the feature variables and the values the extracted features
    :returns labels_ros: the resampled labels through the oversampling of the minority class
    :type labels_ros: list of strings
    :returns lines_ros: the resampled rows of the dataset
    :type lines_ros: a list of lists
    '''
    #we oversample the minority label
    labels = [line[-1] for line in lines]
    #taken from https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html on 2022/03/08
    lines_ros, labels_ros = ros.fit_resample(lines, labels)

    features = []
    #we extract features and labels
    feature_value = {}
    for line in lines_ros:
        for feature_name in selected_features:
            line_index = feature_to_index.get(feature_name)
            feature_value[feature_name] = line[line_index]
        features.append(feature_value)
    return features, labels_ros, lines_ros

#This function is taken and adjusted from the course Machine Learning for NLP on 2022/03/07
def extract_embeddings(lines, word_emb_model):
    '''
    Function that extract word embeddings from the rows of a dataset.

    :param lines: the rows of a dataset
    :type lines: list of lists
    :param word_emb_model: the embedding model used to extract the embeddings from the tokens of each row

    :returns embeddings: the embedding vectors of the dataset
    :type embeddings: a list of arrays / matrix
    '''
    embeddings = []
    for line in lines:
        if line[0] in word_emb_model:
            vector = word_emb_model[line[0]]
        else:
            vector = [0]*300
        embeddings.append(vector)
    return embeddings

#This function is taken and adjusted from the course Machine Learning for NLP on 2022/03/07
def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''
    Function that takes sparse and dense feature representations and appends their vector representation

    :param dense_vectors: list of dense vector representations
    :param sparse_features: list of sparse vector representations
    :type dense_vector: list of arrays
    :type sparse_features: list of lists

    :returns: list of arrays in which sparse and dense vectors are concatenated
    '''

    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())

    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
    return combined_vectors

def train_apply_evaluate(desired_features, training, testing, embeddings = False, language_model = None):
    '''
    Function that trains (SVM), applies and evaluates a classifier for the predicate prediction task.

    :param desired_features: the features used for the predicate prediction task to be extracted from the datasets
    :type desired_features: a list of strings, where each string represents a feature variable
    :param training: the training rows
    :type training: a list of lists
    :param testing: the test rows
    :type testing: a list of lists
    :keyword param embeddings: if set to False(default) the model is not going to use word-embeddings as features and the linear kernel will be used.
                               if set to True word-embeddings will be extracted and combined with sparse vectors and the gaussian kernel will be used.
    :type emebeddings: boolean
    :keyword param language_model: set to None by default bc embeddings are not used as features. It needs to be defined when embeddings param is set to True

    :returns predictions: the prediction labels of the classification
    :type predictions: a list of strings, where each string is a label
    '''                          
    vectorizer = DictVectorizer()

    train_features, train_labels, train_lines_ros = extract_features_and_gold_labels(training, desired_features)
    train_features_vec = vectorizer.fit_transform(train_features)

    test_features, test_labels, test_lines_ros = extract_features_and_gold_labels(testing, desired_features)
    test_features_vec = vectorizer.transform(test_features)

    #if we are using word embeddings as features
    if embeddings == True:
        train_emb = extract_embeddings(train_lines_ros, language_model)
        test_emb = extract_embeddings(test_lines_ros, language_model)
        #We combine the embeddings with the previously extracted features
        train_features_vec = combine_sparse_and_dense_features(train_emb, train_features_vec)
        test_features_vec = combine_sparse_and_dense_features(test_emb, test_features_vec)

        used_clf = SVC(kernel = 'rbf')#, gamma=0.01, class_weight = 'balanced')
    else:
        used_clf= SVC(kernel = 'linear')#, gamma=0.01, class_weight = 'balanced')

    #We train the classifier
    clf = used_clf.fit(train_features_vec, train_labels)
    #We apply the classifier
    predictions = clf.predict(test_features_vec)

    #We evaluate
    report = classification_report(test_labels, predictions, zero_division = False, digits = 7)
    print(report)

    return predictions
