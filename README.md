# BayesMC:Parameter estimation using Bayesian inference with Markov Chain Monte Carlo


# File download 
1. Download **Dataset** and **Results** directory from https://osf.io/jcqke/?view_only=6c63f45e6097491e9625688aa816b794.
2. Put the two folders inside the directory where the rest of the code it. Code won't run until you download this two folder

Example: 
If **BayesMC** is the main directory where you want to put everything, then **BayesMC** should have the following files/Folders

1. Dataset
2. Experimental_data
3. Results
4. bayes.py
5. de.py
6. de_snooker.py
7. h5.py
8. NN_training.py
9. run.py

# How to run and save results

The code runs from **run.py** and saves all results from the current run in a subfolder with a time stamp inside the **Results** folder.
During every run the following files/folders are generated in a :
1. "**_timestamp__simname_**_trained_model.h5" stores the weights and biases trained NN model.
2. "**_timestamp__simname_**_train_test.h5" stores the data used for training and testing the NN netwrok.
3. "**_timestamp__simname_**_scaler.joblib" stores the scaler used to perform transformation of the training data.
4. A new folder is created "**_timestamp__simname_**_bayes" which saves the corner plot as well as the MCMC chains generated from the current run.

A zip file, **20220826-040829one_diode.zip**, with sample results have been provided in **Results**. Unzip the folder to see the file structure.

# Steps to run the code without training the NN
It is possible to run the code without training the NN surrogae model. A pre-trained NN has been provided in **20220826-040829one_diode.zip**
1. Unzip the zip file **20220826-040829one_diode.zip**
2. The folder **20220826-040829one_diode.zip** contains 3 files required to run a the code without training the NN surrogate model:

    (a) A hdf5 file **20220826-040829one_diode_trained_model.h5** which has the weight and biases of the trained model.

    (b) A hdf5 file **20220826-040829one_diode_trained_model.h5** which contains the train and test data needed to verify if the NN is working correctly.

    (c) A scaler file **20220826-040829scaler.joblib** containing the scalers used for data transformation, required to inverse transform the data at the end of the run.
3. Open the **run.py** file in an editor and change **_nn_trained_** to **yes**. The _elif_ statement will be executed. The Paths to the files for this have already been set to the three files mentioned above for the code to run without training the NN.



