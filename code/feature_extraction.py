#!/usr/bin/env python
# coding: utf-8


from multiprocessing.sharedctypes import Value
import spacy
nlp = spacy.load("en_core_web_sm")
import sys

def get_path(sent_data, current_token_index, predicate_index, path_length):
    """
    Recursive function that returns for a given token and predicate wether the former is a descendant of the latter,
    and if so the length of the path between the two. If the token is NOT a descendent of the latter, the path_length
    that is returned is 100.
    :param sent_data: a list of all the data of the current sentence
    :type sent_data: list of lists, where each list contains sentence data
    :param current_token_index: the index in the sentence of the current token
    :type current_token_index: int
    :param predicate_index: the index in the sentence of the predicate we are currently considering
    :type predicate_index: int
    :path_length: the number of steps we have currently taken to reach the predicate from the token
    :type path_length: int
    :returns descendant: binary feature, whether the token is a descendant of the path (1) or not (0)
    :type descendant: int (o or 1)
    :returns length : the number of steps it takes to reach the predicate from the token - 100 if there is no direct path
    :type length: int
    """
    row = sent_data[int(current_token_index)]
    try:
        current_head = int(row[HEAD])
    except ValueError: # if the head is not defined, return false
        return 0, 100
    #return true if the predicate is the current row, otherwise take another step on the path
    if int(current_token_index) == int(predicate_index):
        return 1, path_length
    #if we reach the root we cannot take another step and must return false
    elif current_head == 0:
        return 0, 100
    else:
        try:
            descendant, length = get_path(sent_data, current_head -1 , predicate_index, path_length + 1 )
            return descendant, length
        except RecursionError: # return false if we get stuck in a loop
            return 0, 100


def extract_extra_features(sentence_string):
    """
    This function extracts the following features for each token in a sentence:
    get extra features: 1. token_index, 2. token, 3. previous token 4. pos , 5. previous pos 6. lemma
    7. dependency path 8.dependency head 9.dependency children
    param sentence_string: a sentence string to extract features
    type sentence_string: string
    returns sent_features: a list of lists with the features of all the tokens in the sentence
    """
    sent_features = []
    doc = nlp(sentence_string)
    for sent in doc.sents: # code from group 7 Konstantina Andronikou, Lahorka Nikolovski, Mira Reisinger
        for token_index, token in enumerate(sent): # adapted code from group 7 Konstantina Andronikou, Lahorka Nikolovski, Mira Reisinger
            if token_index >0 :
                pre_token = sent[token_index-1]
                pre_pos = pre_token.pos_
            else:
                pre_token = "<s>"
                pre_pos = "<s>"

            extra_features = [str(token.lemma_),
                              str(token.pos_),
                              str(token.tag_),
                              str(token.dep_),
                              str(token.head.text),
                              str(pre_token),
                              str(pre_pos)]

            sent_features.append(extra_features)
    return sent_features # a list



def get_gold_predicates_and_data(sent_data, sent_index):
    ''''
    This function makes sure to extract all the relevant features for a given input sentence and returns this data
    param sent_data: all the data from the current sentence
    type sent_data : list of lists, with in each list the data for one token of the sentence
    param sent_index: the index of the current sentence
    type sent_indes: int
    :returns sent_rows_for_predicate: the extracted information of the sentence per predicate per token
    :type sent_rows_for_predicate:  list of lists, with each list contraining strings
    :returns sentence_string: a list of all comlete sentence strings in the data
    :type sentence_strings: a list of strings
    '''

    sentence_string = " ".join([row[TOKEN] for row in sent_data])
    #extract features for the tokens in the sentence
    extracted_features_data = extract_extra_features(sentence_string)

    #count the number of predicates
    try:
        if sent_data[0][11] == '':  #via Microsoft Watch from Mary Jo Foley
            return False, False
        n_predicates = int(len(sent_data[0]) - 11)
    except:
        #skip empty instances: cases where there is no predicate in the sentence (the length is smaller than eleven)
        return False, False

    sent_rows_for_predicate = []

    #iterate over all predicated in the sentence
    for i in range(n_predicates):
        #get the column index of the column in which the information for the current predicate is located
        predicate_column_index = int(11 + i)
        #extract the row of the predicate
        try:
            current_predicate_row = [row for row in sent_data if 'V' == row[predicate_column_index]][0]
        except IndexError: #in case there is no predicate to be found on this column index
            continue
        if len(current_predicate_row) == 0 : #if the predicate row is empty we go on to the next predicate
            continue

        # EXTRACT PREDICATE INFO

        #predicate lemma
        predicate_lemma = current_predicate_row[LEMMA]
        #predicate position in the sentence; -1 because python counts from 0 and the dataset from 1
        predicate_position = int(current_predicate_row[TOKEN_INDEX]) -1
        #pos of the predicate (from spacy)
        predicate_pos = extracted_features_data[int(predicate_position)-1][1]
        #pos tag of the predicate (from spacy)
        predicate_postag = extracted_features_data[int(predicate_position)-1][2]
        #dependency label tag of the predicate (from spacy)
        predicate_dep = extracted_features_data[int(predicate_position)-1][5]

        #ITERATE THE GOLD DATA
        for i, row in enumerate(sent_data):
            try:
                token_index = int(row[TOKEN_INDEX]) -1
            except ValueError: #if token indices are used that are NOT integers, skip this sentence
                return False, False
            if token_index == predicate_position:
                continue
            #extract whether the current token is a descendent of the current predicate and how long the path is
            predicate_descendant, path_length = get_path(sent_data, i, predicate_position, 0)

            #save data
            token_data =[str(sent_index), str(token_index), row[TOKEN], str(predicate_descendant), str(path_length)]
            #add extracted data
            token_data.extend(extracted_features_data[i])
            #add predicate data
            token_data.extend([predicate_lemma, str(predicate_position), predicate_pos, predicate_postag, predicate_dep, row[predicate_column_index]])
            sent_rows_for_predicate.append(token_data)
        #all_predicate_info.append(predicate_info)
    return sent_rows_for_predicate, sentence_string



def clean_and_rearrange_data(inputfile):
    ''''
    This function reads in the data, extracts additional information on the tokens and restructures it in the following way:
    the rows in the dataset are the tokens of the setences. Each sentence token gets as many rows as there are
    predicates in the sentence. So if there are 5 tokens in the sentence, and 2 predicates, this sentence will be represented
    over 5 * 2 = 10 rows
    :param inputfile: path to file containing conll data
    :type inputfile: string
    :returns all_data: all the restructured, extracted data
    :type all_data: list of lists, where each list contains all extracted information for one token in a sentence for
                    one specific predicate
    :returns sent_strings: all the full sentence strings of the sentences in the input data
    :type sent_strings: list of strings
    '''
    with open(inputfile, 'r', encoding='utf8') as infile:
        rows = infile.readlines()

    sent_index = 0
    #a list of lists, where each list corresponds to one datapoint
    sent_rows = []
    sent_strings = []
    all_data = []
    for row in rows:
        #skip comments
        if row[0] == '#':
            continue

        #once we reach the new line char, that means we are at the end of the sentence, so we process everything inside the temp_sent_data_list
        if row == '\n':
            predicate_sent_rows, sent_string =  get_gold_predicates_and_data(sent_rows, sent_index)

            if predicate_sent_rows == False:
                sent_rows = []
                continue

            all_data.extend(predicate_sent_rows)
            sent_strings.append((sent_index, sent_string))
            #predicates_metadata.append(predicate_metadata)
            #all_data_features.extend(extracted_sent_features)
            sent_index += 1
            sent_rows = []
            continue
        #split row into a list
        datapoint = row.strip('\n').split('\t')
        #add current row to our sentence rows
        sent_rows.append(datapoint)

    return all_data, sent_strings



def store_to_file(input_list, headers_list, output_file):
    """
    This function saves a dataset as a list of lists to a conll file, where each list becomes a datarow, and
    different entries in the lists are separated by tabs
    :param input_list : the data to be stored
    :type input_list : a list of lists (of strings) [[str,]]
    :param headers_list: the headers of the columns in the data
    :type headers_list: a list of strings
    :param output_file: the path to the file in which the data should be stored
    :type output_file: string
    """
    with open(output_file, 'w', encoding="utf-8") as outfile:
        headers = '\t'.join(headers_list) + '\n'
        outfile.write(headers)

        for lst in input_list:
            data_row = "\t".join(lst) + '\n'
            outfile.write(data_row)



def main(argv=None):
    """
    This code in this file takes in a conll dataset input file of SRL data, extracts relevant features from this data,
    restructues the data so each row represents one token in a sentence, where each tokens gets a separate entry for
    all the predicates in the sentence. The goal of restructuring the data like this is that the new dataset can be used
    for predicate extraction and argument classification
    :param my_arg : a list containing the following parameters:
                    args[1] : the path (str) to the conll input data
                    args[2] : the path (str) to the conll output data - the location where the new dataset should be stored
    """
    global TOKEN_INDEX, TOKEN, LEMMA, HEAD, DEPENDENCY_LABEL
    TOKEN_INDEX = 0
    TOKEN = 1
    LEMMA = 2
    HEAD = 6
    DEPENDENCY_LABEL = 7
    if argv is None:
        argv = sys.argv
    inputfile = argv[1]
    outputfile = argv[2]
    print(inputfile, outputfile)
    headers = ['sent_index', 'token_index', 'token', 'predicate_descendant', 'path_to_predicate_length',
            'lemma', 'pos', 'postag', 'dependency', 'head_text','prev_token', 'prev_pos', 'predicate_lemma', \
            'predicate_index', 'predicate_pos', 'predicate_postag', 'predicate_dependency', 'argument']
    dev_processed, dev_sents = clean_and_rearrange_data(inputfile)
    store_to_file(dev_processed, headers, outputfile)




if __name__ == '__main__':
    main()
