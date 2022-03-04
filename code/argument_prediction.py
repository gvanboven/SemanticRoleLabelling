#import models
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm

import sys
import numpy as np

#import gensim to deal with word embeddings
import gensim
    
def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row
    
    :param row: row from conll file
    :param selected_features: list of selected features
    
    :type row: string
    :type selected_features: list of strings

    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    #only extract selected features
    for feature_name in selected_features:
        #get index of current feature to extract it
        r_index = feature_to_index.get(feature_name)
        feature_value = row[r_index]
        if feature_name in integer_features:
            feature_value = int(feature_value)
        feature_values[feature_name] = feature_value
        
    return feature_values

def extract_word_embedding(token, word_embedding_model):
    '''
    Function that returns the word embedding for a given token out of a distributional semantic model.
    If the token is not present in the embeddings model, a 300-dimension vector of 0s is returned.
    
    :param token: the token
    :param word_embedding_model: the distributional semantic model
    :type token: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    
    :returns a vector representation of the token
    '''
    if token in word_embedding_model:
        vector = word_embedding_model[token]
    else:
        vector = np.zeros(300)
    return vector

def create_vectorizer_traditional_features(feature_values):
    '''
    Function that creates vectorizer for set of feature values, and vectorizes these
    
    :param feature_values: list of dictionaries containing feature-value pairs
    :type feature_values: list of dictionairies (key and values are strings)
    
    :returns vectorizer: vectorizer fitted on feature values
    :returns vec_feature_values: vectorized feature values
    '''
    vectorizer = DictVectorizer()

    vec_feature_values = vectorizer.fit_transform(feature_values)
    return vectorizer, vec_feature_values

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
    
    #concatenate sparse and dense vectors
    for index, vector in enumerate(sparse_vectors):
        combined_vector = np.concatenate((vector,dense_vectors[index]))
        combined_vectors.append(combined_vector)
    return combined_vectors

def extract_embeddings_as_features_and_gold(conllfile, word_embedding_model, selected_features, n_rows):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :param selected_features: the features to include in the models
    :param n_rows: the number of rows that should be included in the model. If 'all', the entire dataset is considered
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type selected_features: list
    :type n_rows: int or 'all'
    
    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    :return vectorizer: vectorizer fitted on feature values
    '''
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    labels = []
    features = []
    embedding_features = []
    sparse_features = []
    
    #read inputfile
    with open(conllfile, 'r', encoding='utf8') as infile:
        rows = infile.readlines()
    if n_rows != 'all': 
        rows = rows[:n_rows]
    firstline = True
    for i, row in enumerate(rows):
        
        row_data = row.strip('\n').split('\t')
        #skip the first line, as this contains the headers of the columns
        if firstline:
            firstline = False
            continue
        sparse_feature_names = [feature for feature in selected_features if feature not in embedding_feature_names]

        #get the embeddings of all features that should get word embedding representations
        embedding_vectors = [extract_word_embedding(row_data[feature_to_index.get(dense_feature)], word_embedding_model) \
                                for dense_feature in embedding_feature_names]
        #concatenate the seperate embeddings
        embeddings = np.concatenate((embedding_vectors))

        #extract other features, store the feature values in a dict
        feature_dict = extract_feature_values(row_data, sparse_feature_names)
        #save embeddings and sparse feature values
        embedding_features.append(embeddings)
        sparse_features.append(feature_dict)
        #store gold argument label
        labels.append(row_data[feature_to_index.get('argument')])
    print('data extracted')
    
    #vectorize sparse features
    vectorizer, sparse_vec_features = create_vectorizer_traditional_features(sparse_features)
    print('sparse features vectorized')

    #combine embeddings and sparse features
    features = combine_sparse_and_dense_features(embedding_features, sparse_vec_features)
    return features, labels, vectorizer

def extract_embeddings_as_features(conllfile, vectorizer, word_embedding_model, selected_features, n_rows):
    '''
    Function that extracts features and gold labels using word embeddings
    
    :param conllfile: path to conll file
    :param vectorizer: vectorizer fitted on training data
    :param word_embedding_model: a pretrained word embedding model
    :param selected_features: the features to include in the model
    :param n_rows: the number of rows that should be included in the model. If 'all', the entire dataset is considered
    :type conllfile: string
    :type vectorizer: sklearn.feature_extraction._dict_vectorizer.DictVectorizer
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors
    :type selected_features: list
    :type n_rows: int or 'all'
    
    :return features: list of vector representation of tokens
    '''
    features = []
    embedding_features = []
    sparse_features = []

    #read inputfile
    with open(conllfile, 'r', encoding='utf8') as infile:
        rows = infile.readlines()
    if n_rows != 'all': 
        rows = rows[:n_rows]
    firstline = True
    for i, row in enumerate(rows):
        row_data = row.strip('\n').split('\t')
        #skip the first line, as this contains the headers of the columns
        if firstline:
            firstline = False
            continue

        sparse_feature_names = [feature for feature in selected_features if feature not in embedding_feature_names]

        #get the embeddings of all features that should get word embedding representations
        embedding_vectors = [extract_word_embedding(row_data[feature_to_index.get(dense_feature)], word_embedding_model) \
                                for dense_feature in embedding_feature_names]
        #concatenate the seperate embeddings
        embeddings = np.concatenate((embedding_vectors))

        #extract other features, store the feature values in a dict
        feature_dict = extract_feature_values(row_data, sparse_feature_names)

        #save embeddings and sparse feature values
        embedding_features.append(embeddings)
        sparse_features.append(feature_dict)
    #vectorize sparse features
    sparse_features_vectors = vectorizer.transform(sparse_features)

    #combine embeddings and sparse features
    features = combine_sparse_and_dense_features(embedding_features, sparse_features_vectors)
    return features

    
def create_classifier(train_features, train_targets):
    '''
    Function that creates a SVM classifier trained on the training data
    
    :param train_features: vectorized training features
    :param train_targets: gold labels for training

    :type train_features: list
    :type train_targets: list of strings (gold labels)
    
    :return model: trained classifier
    '''

    model = svm.LinearSVC()

    #train model
    model.fit(train_features, train_targets)
    return model
    
    
def classify_data(model, test_features, testfile, outputfile, n_rows):
    '''
    Function that makes prediction on the test data, given a trained model and saves the predictions to the outputfile
    
    :param model: trained classifier
    :param test_features: vectorized test features
    :param inputdata: path to conll file containing test data
    :param outputfile: path to conll output file

    :type model: sklearn model
    :type test_features: list
    :type inputdata: string
    :type outputfile: string
    '''

    #make predictions
    predictions = model.predict(test_features)

    #save predictions 
    outfile = open(outputfile, 'w')
    counter = 0

    with open(testfile, 'r', encoding='utf8') as infile:
        rows = infile.readlines()

    if n_rows != 'all': 
        rows = rows[:n_rows]

    for line in rows:
        #add header to the newly added column of predictions
        if counter == 0:
            new_colum = 'prediction'
            outfile.write(line.rstrip('\n') + '\t' + new_colum  + '\n')
            counter += 1
        else:
            #add predictions in a new column
            if len(line.rstrip('\n').split()) > 0:
                outfile.write(line.rstrip('\n') + '\t' + predictions[counter-1] + '\n')
                counter += 1
    outfile.close()

def main(argv=None):
    '''
    Main function that creates and trains a model on provided training data, classifies the test data, and saves the predictions in an outputfile, which
    contains the inputfile-content plus an additional column in which the predictions are stored.

    :param argv : a list containing the following parameters:
                    mandatory:
                    argv[1] : the path (str) to the conll training data 
                    argv[2] : the path (str) to the conll test data 
                    argv[3] : path (str) to output conll file in which the predictions will be stored
                    argv[4]: number of datarows to consider in training and testing, can be used to train on smaller portions of the data (e.g. to test the system)
                            this should be an int. If all rows should be considered, the input should be 'all'

                    optional:
                    argv[5] : a string of selected features (separated by spaces), to test a combination of features different from the settings
                    
    '''
    #defines the column in which each feature is located
    global feature_to_index, embedding_feature_names, integer_features

    feature_to_index = {'token_index':1, 'token':2, 'head_lemma':3, 'predicate_descendant': 4,
            'path_to_predicate_length':5,  'pos':7, 'postag':8, 'prev_token':9, 'prev_pos': 10, 'dependency':11, 
            'head_text':12, 'predicate_lemma':13, 'predicate_index':14, 'predicate_pos':15, 
            'predicate_dependency': 17, 'argument':18}
    embedding_feature_names = ['token', 'head_lemma', 'prev_token', 'head_text']
    integer_features = ['token_index', 'path_to_predicate_length', 'predicate_index']

    #picking up commandline arguments
    if argv is None:
        argv = sys.argv

    print(argv)

    trainingfile = argv[1]
    testfile = argv[2]
    outfile_path = argv[3]
    n_rows = argv[4]
    try:
        n_rows = int(n_rows)
    except:
        n_rows = 'all'

    #if features to include and are defined, extract them. otherwise use standard values (include all features)
    try:
        selected_features = argv[5].split()
        print(f"features that will be considered in the model are : {selected_features}")
    except:
        selected_features = list(feature_to_index.keys()) #if the selected features are not defined, use all of them
        selected_features.remove('argument') # but of course take out the gold label 


    print(f"start loading in data")

    #extract training and test data
    ## load word embeddings
    language_model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin.gz', binary=True)
    #extract train and test featers and gold labels
    print('training features')
    training_features, gold_labels, vec = extract_embeddings_as_features_and_gold(trainingfile, language_model, selected_features, n_rows)
    print('test features')
    test_features =  extract_embeddings_as_features(testfile, vec, language_model, selected_features, n_rows)

    print(f"start training SVM model")
    #build and train model
    ml_model = create_classifier(training_features, gold_labels)
    print('done training SVM model. Will now start classifying training instances')
    #classify test data and save the predictions
    classify_data(ml_model, test_features, testfile, outfile_path, n_rows)
    print('done classifying')
    
if __name__ == '__main__':
    main()
