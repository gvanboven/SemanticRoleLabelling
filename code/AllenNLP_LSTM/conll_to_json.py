import jsonlines

#the paths to the data in our repository. Change the paths if you want to convert files in different locations
TRAINFILE = "../../data/srl_univprop_en.train.conll" 
DEVFILE = "../../data/srl_univprop_en.dev.conll"
TESTFILE = "../../data/srl_univprop_en.test.conll"

#the output paths to where our data will be stored. 
# Change the paths if you want the store the files at a different location or with a different name
train_outfile = "./data/train.jsonl"
dev_outfile = "./data/dev.jsonl"
test_outfile = "./data/test.jsonl"

TOKEN_INDEX = 0
TOKEN = 1 
LEMMA = 2
POSTAG = 4
PREDICATE_INDEX = 10

def get_sentence_predicate_data(sent_data):
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

    sentence_tokens = [row[TOKEN] for row in sent_data]

    #count the number of predicates
    try:
        if sent_data[0][11] == '':  #via Microsoft Watch from Mary Jo Foley
            return False
        n_predicates = int(len(sent_data[0]) - 11)
    except:
        #skip empty instances: cases where there is no predicate in the sentence (the length is smaller than eleven)
        return False
        
    sentence_predicate_data = []

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

       # EXTRACT GOlD PREDICATE INFO

        #predicate position in the sentence; -1 because python counts from 0 and the dataset from 1
        predicate_position = int(current_predicate_row[TOKEN_INDEX]) -1
        #
        predicate_sense_label = current_predicate_row[PREDICATE_INDEX]
        #
        predicate_label = current_predicate_row[predicate_column_index]
        #
        predicate_postag = current_predicate_row[POSTAG]

        argument_labels = [row[predicate_column_index] for row in sent_data]

        BIO_argument_labels = ["B-"+ argument if argument != '_' else "O" for argument in argument_labels ]

        predicate_data_entry = {"seq_words" : sentence_tokens,
                                "BIO" : BIO_argument_labels,
                                "pred_sense" : [predicate_position, predicate_sense_label, predicate_label, predicate_postag]}
        sentence_predicate_data.append(predicate_data_entry)
    
    return sentence_predicate_data


def clean_and_rearrange_data(inputfile, outputfile):
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
        
    #a list of lists, where each list corresponds to one datapoint
    sent_rows = []
    all_data = []

    for row in rows:
        #skip comments
        if row[0] == '#':
            continue
        
        #once we reach the new line char, that means we are at the end of the sentence, so we process everything inside the temp_sent_data_list
        if row == '\n':
            predicate_sent_data =  get_sentence_predicate_data(sent_rows) 
            if predicate_sent_data == False:
                sent_rows = []
                continue
                
            all_data.extend(predicate_sent_data)
            sent_rows = []
            continue
        #split row into a list
        datapoint = row.strip('\n').split('\t')
        #add current row to our sentence rows
        sent_rows.append(datapoint)
        
    with jsonlines.open(outputfile, 'w') as writer:
        writer.write_all(all_data)

clean_and_rearrange_data(TRAINFILE, train_outfile)
clean_and_rearrange_data(DEVFILE, dev_outfile)
clean_and_rearrange_data(TESTFILE, test_outfile)