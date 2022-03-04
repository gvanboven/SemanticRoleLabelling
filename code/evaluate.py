from collections import defaultdict, Counter
import sys
import pandas as pd

#open data
def extract_annotations(inputfile: str, annotationcolumn: str, delimiter: str ='\t'):
    '''
    This function extracts annotations represented in the conll format from a file
    
    :param inputfile: the path to the conll file
    :type inputfile: string
    :param annotationcolumn: the index of the column in which the target annotation is provided
    :type annotationcolumn: int 
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type delimiter: string

    :returns annotations: the extracted annotations 
    :type annotations: a list of strings
    '''
    annotations = []
    first_line = True
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            #skip the first line in the data, this contains the column names (which equal the indices)
            if first_line:
                first_line = False
                continue
            components = line.rstrip('\n').split(delimiter)
            #skip empty lines
            if len(components) > 1:
                annotations.append(components[int(annotationcolumn)])
    return annotations

def obtain_counts(gold_annotations, machine_annotations):
    '''
    This function counts the number of correct and incorrect machine predictions for each class in the gold annotations
    
    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: a list of strings
    :type machineannotations: a list of strings
    
    :returns evaluation counts: a countainer providing the counts for each predicted and gold class pair
    :type evaluation counts: a default dict that lookst as follows: {gold annotation : {machine annotation : count}}
    '''
    evaluation_counts = defaultdict(Counter)
    for i, annotation in enumerate(gold_annotations):
        evaluation_counts[annotation][machine_annotations[i]] += 1
    return evaluation_counts
        
def safe_divide(numerator, denominator):
    """
    This function divides the numerator by the denominator. If the denominator is zero it returns zero
    :type numerator: int or float
    :type denominator: int or float
    :returns type: float or int
    """
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0
    
def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate the precision, recall and micro/macro F1-score for each class and returns the class specific scores
    in a dictionary. The overall performance scores of the system are printed 
    
    :param evaluation_counts: a dict containing counts for each gold annotation class - machine annotation class combination
    :type evaluation_counts: a default dict that looks as follows: {gold annotation : {machine annotation : count}}
    
    :returns scores : the precision, recall and f-score of each class in a container
    :type scores: a dict with the scores per class {class: {'precision' : float, 'recall': 'float', 'F1': float}}
    '''
    scores = {}
    #count TP, TP+FP+ TP+FN counts to compute the micro F1
    all_TP, all_TPFP, all_TPFN = 0,0,0 
    for classlabel, counts in evaluation_counts.items():
        #True positives are the number of times we correctly classify the label
        TP = counts[classlabel]
        #false negatives are the number of times we should have selected the current label but selected another
        FN = sum([count for label, count in counts.items() if label != classlabel])
        #false positives are the number of times we should have selected another label but selected this label
        FP = sum([label_counts[classlabel] for label, label_counts in evaluation_counts.items() if label != classlabel])
        
        #save TP FP and FN counts to compute micro F1
        TPFP = TP + FP
        TPFN = TP + FN
        all_TP += TP
        all_TPFP += TPFP
        all_TPFN += TPFN

        #calculate metrics, make sure to safe divide in case the denominator is zero
        precision = round(safe_divide(TP, (TP + FP)),3)
        recall = round(safe_divide(TP, (TP + FN)),3)
        F1 = round(safe_divide((2* precision * recall), (precision + recall)),3)
        #save scores
        scores[classlabel] = {'precision' : precision, 'recall': recall, 'f-score': F1}
        
    #get marco (unweighted) averages    
    macro_precision = sum([score['precision'] for score in scores.values()]) / len(scores.keys())  
    macro_recall = sum([score['recall'] for score in scores.values()]) / len(scores.keys())  
    macro_F1 = safe_divide((2* macro_precision * macro_recall), (macro_precision + macro_recall))

    #get mirco (unweighted) averages   
    micro_precision = all_TP / all_TPFP
    micro_recall = all_TP / all_TPFN
    micro_F1 = safe_divide((2* micro_precision * micro_recall), (micro_precision + micro_recall))
    
    #print macro and micro averages
    print("\nOverall performance scores:\n")
    print(f"Macro precision score : {round(macro_precision * 100,2)}")
    print(f"Macro recall score : {round(macro_recall * 100,2)}")
    print(f"Macro F1 score : {round(macro_F1 *100,2)}\n")
    print(f"Micro F1 score : {round(micro_F1 *100,2)}")
    return scores

def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class
    
    :param evaluation_counts: a dict containing counts for each gold annotation class - machine annotation class combination
    :type evaluation_counts: a default dict that looks as follows: {gold annotation : {machine annotation : count}}

    :prints out a confusion matrix
    '''
    #make sure all values are in the dict, and that the same order is maintained for all labels, so that we get a clean table
    for i in evaluation_counts.keys():   
        evaluation_counts[i] = {j:evaluation_counts[i][j] for j in evaluation_counts.keys()}
        
    # create matrix
    confusions_pddf = pd.DataFrame.from_dict({i: evaluation_counts[i]
                                              for i in evaluation_counts.keys()},
                                             orient='index', columns=evaluation_counts.keys(),
                                             )
    #print matrix and latex version of matrix
    print("\nConfusion matrix:\n")
    print(confusions_pddf)


def carry_out_evaluation(inputfile, goldcolumn_index, systemcolumn_index, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param inputfile: path to conll file with gold labels and system annotations
    :type inputfile: string
    :param goldcolumn_index: indication of column with gold annotations
    :type goldcolumn_index: int
    :param systemcolumn_index: indication of column with system predictions
    :type systemcolumn_index: int
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    :type delimiter: string
    
    :returns evaluation_outcome: evaluation information for this specific system
    :type evaluation_outcome: dict with performance scores per class {class: {'precision' : float, 'recall': 'float', 'F1': float}}
    '''
    #extract gold annotations
    gold_annotations = extract_annotations(inputfile, goldcolumn_index)
    #retrieve annotations of the system
    system_annotations = extract_annotations(inputfile, systemcolumn_index, delimiter)
    #evaluate
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    
    #print confusion matrix
    provide_confusion_matrix(evaluation_counts)
    #get evaluation metrics
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)
    return evaluation_outcome

def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems
    
    :param evaluations: the outcome of evaluating one or more systems
    :type evaluations:  dict with performance scores per class {class: {'precision' : float, 'recall': 'float', 'F1': float}}
    '''
    evaluations_pddf = pd.DataFrame.from_dict(evaluations, orient='index')
    print("\nperformance scores per class:\n")
    print(evaluations_pddf)


def main(my_args=None):
    '''
    This main function makes sure to carry out the evaluation of a conll file wich contains both the gold labels and 
    the machine annotations. This file prints out the confusion matrix and the overall macro precision, recall and F1, 
    as well as the micro F1 score.
    Aditionally, the precision, recall and F1 are computed per class, and a table is printed out that gives an overview of 
    these scores. 
    
    :param my_arg : a list containing the following parameters:
                    args[1] : the path (str) to the conll inputfile
                    args[2] : the index (int) of the column in the inputfile in which the gold labels can be found
                    args[3] : the index (int) of the column in the inputfile in which the machine annotations can be found
    '''
    if my_args is None:
        my_args = sys.argv
    
    inputfile = my_args[1]
    y_true_index = my_args[2]
    y_pred_index = my_args[3]
 
    evaluations = carry_out_evaluation(inputfile, y_true_index, y_pred_index)
    provide_output_tables(evaluations)

if __name__ == '__main__':
    main()