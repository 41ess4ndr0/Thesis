In this foldere there are many python files:
* 1_dataframe_split.py  -> file for sampling 50000 units and split it into test and training
* 2 training_v1.py (OLD VERSION)-> training either the model of gpt2 or already pre trained one for a certain number of samples 
* 2 training_v2.py (NEW VERSION)-> training either the model of gpt2 or already pre trained one for a certain number of samples
* 3 generating.py -> for the generation of patent abstract in the test set
* 4 test.py -> testing the generated patent abstract through "3 generating.py"
* 5 synthetic patent.py -> use the first method though a loop for generating the synthetic abstract with some input we decide
* 6 synthetic patent_v2.py -> use the second method though the entire generation for the synthetic abstract with some input we decide
* 7 xi_real_instruction.py -> we tested the entire engine for the xi_real estimation
* 8 test set generation.py -> trial test for the generation of patent abstracts
* 9 hyperparameter tuning.py -> for the tuning of the paramenter TOP_P and TEMPERATURE
* 10 2020_test_similarity_xi_real.py -> for the results of the out of sample 2020ù
* utils -> contains the main functions used inside the codes
