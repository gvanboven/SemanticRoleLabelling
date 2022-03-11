In this folder you find the code needed to implementing and training the AllenNLP LSTM SRL model. 

The file `conll_to_json.py` can be used to transform `conll` input data to `jsonl` data, in the format that the AllenNLP model expects it. We have already transformed our input data, and stored the new files in the `./data` folder. This file can be run by setting the input and output paths in the file, and running the following command in the terminal:     
`python .\conll_to_json.py`

The file `srl_main.py` can then be used to train and test the model. This code was provided by our teachers Pia Sommerauer, Antske Fokkens and José Angel Daza. In this file, the path to the converted training and test data should be specified, as well as the path to which the model should be stored. We have currently set the paths to the data in our `./data` folder. 