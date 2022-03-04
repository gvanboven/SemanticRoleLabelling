import sys
import argument_prediction
import feature_extraction
import evaluate

def main(argv=None):
    '''
    Main function that takes a train and test conll file for SRL prediction as input and does the following:
    i. extracts features for SRL predicate and argument classification and saves the features in a new conll file. 
        This new dataset will be named '[trainingfilename]_features.conll'
    ii. extracts the predicates of the test sentences using a rule-based method, and evaluates the performance
    iii. trains a SVM classifier to predict the agruments, uses this model to make predictions on the test data and evaluates the predictions

    The predicate and argument predictions will be added to the dataset created in step i) and saved as new datasets. 
    The names of these datasets will be '[trainingfilename]_features_predicate_predictions.conll' and 
    '[trainingfilename]_features_argument_predictions.conll' respectively.

    :param my_arg : a list containing the following parameters:
                    args[1] : the path (str) to the conll training data 
                    args[2] : the path (str) to the conll test data 
                    argv[3]: number of datarows to consider in training and testing, can be used to train on smaller portions of the data (e.g. to test the system)
                            this should be an int. If all rows should be considered, the input should be 'all'
                    
                    optional:
                    argv[4] : a string of selected features (separated by spaces), to test a combination of features different from the settings
                    
    '''
    #picking up commandline arguments
    if argv is None:
        argv = sys.argv

    GOLD_INDEX = -2
    PRED_INDEX = -1

    trainingfile = argv[1]
    testfile = argv[2]
    n_rows = argv[3]

    #if the features to use for argument prediction are defined, extract them
    try:
        selected_features = argv[4].split()
        print(f"features that will be considered in argument prediction are : {selected_features}")
    except:
        selected_features = None

    #create paths to the output files in which the features and predictions should be saved
    train_features_output_path = trainingfile.replace('.conll','_features.conll')
    test_features_output_path = testfile.replace('.conll', '_features.conll')
    predicate_predictions_output_path = test_features_output_path.replace('.conll', '_' + str(n_rows) + '_predicate_predictions.conll')
    argument_predictions_output_path = test_features_output_path.replace('.conll', '_' + str(n_rows) + '_argument_predictions.conll')

    #extract the features of the train and test data
    print("extract train features")
    feature_extraction.main(['', trainingfile, train_features_output_path])
    print("extract test features")
    feature_extraction.main(['', testfile, test_features_output_path])

    #make predicate predictions on the test data and evaluate
    ##TODO: predicate predicions
    #predicate_prediction.main(train_features_output_path, test_features_output_path, predicate_predictions_output_path, n_rows)
    ##TODO: evaluation 
    #evaluate.main(predicate_predictions_output_path, GOLD_INDEX, PRED_INDEX)

    #train argument classifier, create predictions on the test data and evaluate
    print('argument prediction')
    argument_prediction.main(['', train_features_output_path, test_features_output_path, argument_predictions_output_path, n_rows, selected_features])
    print('argument evaluation')
    evaluate.main(['', argument_predictions_output_path, GOLD_INDEX, PRED_INDEX])

if __name__ == '__main__':
    main()