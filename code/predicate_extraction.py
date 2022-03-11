#!/usr/bin/env python
# coding: utf-8


import gensim
import pickle
import sys


print('loading embedding model...')
print()
emb_model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('embedding model loaded successfully')

from predicate_extraction_utils import new_dataset, read_in_conll_file, train_apply_evaluate, get_distribution,                                     rule_based_pred_extraction


#The commented out code is cept for running the file in jupiter notebooks, separate from the pipiline

#train_path = '/Users/vicky/Desktop/Master/Period4/NLP/data/train.conllu'
#train_fixed_path = '/Users/vicky/Desktop/Master/Period4/NLP/data/train.fixed.conllu'
#dev_path = '/Users/vicky/Desktop/Master/Period4/NLP/data/dev.conllu'
#dev_fixed_path = '/Users/vicky/Desktop/Master/Period4/NLP/data/dev.fixed.conllu'
#test_path = '/Users/vicky/Desktop/Master/Period4/NLP/data/test.conllu'
#test_fixed_path = '/Users/vicky/Desktop/Master/Period4/NLP/data/test.fixed.conllu'

#lists_file = '../data/row_lists.pkl'
#taken from https://www.adamsmith.haus/python/answers/how-to-save-and-read-a-list-in-python#:~:text=Use%20pickle.,the%20file%20of%20file_name%20respectively.
#on 2022/03/08
#open_file = open(lists_file, "wb")
#pickle.dump([train_lines,dev_lines,test_lines],open_file)
#open_file.close()

#open_file = open(lists_file, "rb")
#loaded_list = pickle.load(open_file)
#train_lines, dev_lines = loaded_list[0], loaded_list[1]
#open_file.close()


#wanted_features = ['token_id','lemma', 'pos','prev_token', 'prev_pos', 'dep', 'head']
#predictions1 = train_apply_evaluate(wanted_features, train_lines, dev_lines)



#wanted_features = ['lemma', 'pos', 'dep' ]
#predictions2 = train_apply_evaluate(wanted_features, train_lines, dev_lines)


#wanted_features = ['token', 'pos_tag', 'dep']
#predictions3 = train_apply_evaluate(wanted_features, train_lines, dev_lines)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    train_path = argv[1]
    dev_path = argv[2]

    #we define the output files for our fixed datasets
    train_fixed_path = train_path.replace('.conllu', '.fixed.conllu')
    dev_fixed_path = dev_path.replace('.conllu', '.fixed.conllu')

    print('Adjusting the datasets for the predicate prediction task...')
    print()
    #we create new datasets adjusted for the predicate extraction task
    new_dataset(train_path, train_fixed_path)
    new_dataset(dev_path, dev_fixed_path)

    print('Extracting and pruning the rows of the datasets...')
    #we extract the rows from the datasets and we prune them
    train_lines = read_in_conll_file(train_fixed_path)
    dev_lines = read_in_conll_file(dev_fixed_path)


    #We predict the predicates using a rule-based method
    predicate_rows = get_distribution(dev_lines)
    print()
    print()
    print('Accuracy of rule-based predicate prediction:')
    rule_based_pred_extraction(dev_lines)
    print()
    print()

    #We predict the predicates with ML using word_embeddings, PoS labels and dependency labels as features
    print('Applying Machine Learning method for predicate prediction')
    print()
    print('Training and evaluation of SVM classifier...')
    print()
    wanted_features = ['pos', 'dep']
    predictions1 = train_apply_evaluate(wanted_features, train_lines, dev_lines, embeddings = True, language_model = emb_model)



if __name__ == '__main__':
    main()
