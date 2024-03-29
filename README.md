# BayesMC:Parameter estimation using Bayesian inference with Markov Chain Monte Carlo


# File download 
1. Clone the code from GitHub
2. Download **Dataset** and **Results** directory from https://osf.io/jcqke/?view_only=6c63f45e6097491e9625688aa816b794.
3. Put the two folders inside the directory where the rest of the code is. **Code won't run until you download this two folder and put then inside the directory**

Example: 
If **BayesMC** is the main directory where you want to put everything, then **BayesMC** should have the following files/Folders
1. Dataset
2. Experimental_data
3. Results
4. submit.sh (needed for running it on MIT supercloud)
5. bayes.py
6. de.py
7. de_snooker.py
8. h5.py
9. NN_training.py
10. run.py

# Code environment
To run the code one need to create a python environment using the **env_emcee.yml** file.


# How to run and save results
The code runs from **run.py** and saves all results from the current run in a subfolder with a time stamp inside the **Results** folder. All data is aved and graphs corner plot are generated automatically.
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

# Instruction to run it on the MIT Supercloud
1. Transfer all the files to their directory. **You should not be logged into your MIT supercloud account to do this. Just do it from your terminal window.** 
    [1. Dataset
    2. Experimental_data
    3. Results
    4. submit.sh (needed for running it on MIT supercloud)
    5. bayes.py
    6. de.py
    7. de_snooker.py
    8. h5.py
    9. NN_training.py
    10. run.py]
2. Make sure the BayesMC directory has all the RWX permissions. (ls -l will show the permissions each directory has). If not then use **chmod a+rwx BayesMC**. 
3. Also make sure the **Results** also has all the permissions. Use **chmod a+rwx Results** to give permission.
4. syntax to transfer your code : (scp -r **Directory_where_your_code is**/BayesMC **Your_username**@txe1-login.mit.edu:/home/gridsan/**Your_username**/
5. SSH into your supercloud aacount.
6. Navigate to your **BayesMC** directory
7. **To run interactive mode** (if you close the terminal window the code stops)\
    (i) Load environment **module load anaconda/2021a**\
    (ii) Request resources : **LLsub -i -s 20 -g volta:1**\
    (iii) To run : **python run.py**\
    (iv) You can see the progress directly on the terminal window.

8. **To run in batch mode** (even if you close the terminal the code still runs)\
     (i) To run : **sbatch batch.sh**\
     (ii) All the commands to load environment, request resources are mentioned in the batch file **batch.sh**\. A **JOBID**   will be assigned to the submitted job and printed on the terminal window. (For eg. Submitted batch job **19733850**; where **19733850** is the **JOBID**)\
     (iii) Use **LLstat** to see the status of your job. If ST is R then the job is running.\
     (iv) Use **tail -f batch.sh.log-JOBID** (enter your **JOBID**) to see the progress printed on terminal window.\
     (v) To stop printing the progress on the screen : **Ctrl + C**.\
     (vi) You can use LLStat to see if the job is still running or it has finished.\
9. To download the resutls :\
scp -r **Your_username**@txe1-login.mit.edu:/home/gridsan/**Your_username**/BayesMC/Results/**Name_of_your_results_folder** **Directory_where_your_code_is**/BayesMC/Results
