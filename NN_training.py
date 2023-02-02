import numpy
import h5
from pathlib import Path
import os
import joblib
from sklearn.preprocessing import StandardScaler
import datetime
from scipy import interpolate
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Lambda,Conv1D,Conv1DTranspose, Conv2DTranspose, LeakyReLU,Activation,Flatten,Reshape
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from scipy import interpolate
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.factory import get_termination, get_reference_directions, get_algorithm
from pymoo.factory import get_sampling
from pymoo.util.display import SingleObjectiveDisplay
import emcee
import de
import de_snooker
import emcee.state
from sklearn.cluster import KMeans
import emcee.autocorr as autocorr

def create_dir(name, identity):
    '''
    create_dir : creates the folder where the traning results will be stored
            
    Parameters 
    ----------
    name : name of the current training
    identity : unique date-time signature of the training 

    returns
    -------
    dir_path : path to the directory where training results will be stored
    dir_name : root name of all files associated with this simulation
    '''
   
    cur_dir = Path(os.getcwd())/('Results')
    if os.path.exists(cur_dir):
        print("file already exists")    
    else:
        os.mkdir(cur_dir)
    dir_name = identity + name
    dir_path = cur_dir/("%s" % dir_name)
    os.mkdir(dir_path)
    return dir_path , dir_name

def load_dataset_both_exp(loc,exp):
    '''
    load_dataset_2_exp : load y, min, max, and par_mat from dataset file for two different type of experiment
           
    Parameters 
    ----------
    loc : path to the dataset

    returns
    -------
    y_norm : numpy array of min-max scaled value of y-axis of the two experiment joined together
    y1_max : scalar value. maximum of the y-axis values of 1st experiment
    y1_min : scaler value. minimum of the y-axis values of the 1est experiment

    y2_max : scalar value. maximum of the y-axis values of 2nd experiment
    y2_min : scaler value. minimum of the y-axis values of the 2nd experiment

    par_mat : numpy array of all parameter combinations in their original space. (NOT ln transformed or scaler transformed)
    ub : numpy array of upper bound
    lb : numpy.array of lower bound
    n_var : number of parameters
    '''
    dataset = h5.H5()
    dataset.filename = Path(loc)
    dataset.load()

    y1_norm = dataset.root.Y1_norm
    y1_max = dataset.root.Y1_max
    y1_min = dataset.root.Y1_min

    y2_norm = dataset.root.Y2_norm
    y2_max = dataset.root.Y2_max
    y2_min = dataset.root.Y2_min

    par_mat = dataset.root.par_mat
    ub = dataset.root.Input.ub
    lb = dataset.root.Input.lb
    print(ub,lb)
    n_var = len(ub)

    y_norm = numpy.column_stack((y1_norm, y2_norm))

    return y_norm, y1_max, y1_min, y2_max, y2_min, par_mat, ub, lb, n_var

def load_dataset_one_exp(loc, exp):
    '''
    load_dataset_1_exp : load y, min, max, and par_mat from dataset file for one type of experiment
            
    Parameters 
    ----------
    loc : path to the dataset

    returns
    -------
    y_norm : numpy array of min-max scaled value of y-axis of experiment
    y_max : scalar value. maximum of the y-axis values of experiment
    y_min : scaler value. minimum of the y-axis values of experiment
    par_mat : numpy array of all parameter combinations in their original space. (NOT ln transformed or scaler transformed)
    ub : numpy array of upper bound
    lb : numpy.array of lower bound
    n_var : number of parameters
    '''
    dataset = h5.H5()
    dataset.filename = Path(loc)
    dataset.load()
    if exp == 'one':
        y_norm = dataset.root.Y1_norm
        y_max = dataset.root.Y1_max
        y_min = dataset.root.Y1_min
    
    elif exp == 'two':
        y_norm = dataset.root.Y2_norm
        y_max = dataset.root.Y2_max
        y_min = dataset.root.Y2_min

    ub = dataset.root.Input.ub
    lb = dataset.root.Input.lb
    print(ub,lb)
    n_var = len(ub)
    par_mat = dataset.root.par_mat
    return y_norm, y_max, y_min, par_mat, ub, lb, n_var


def par_fit_transform(par_mat):
    '''
    par_fit_transform : perform StandardScaler transformation of the input parameters into the space required by the Neural Netwrok. 
                        before aplying StandardScaler transformation a natural log transform is done on all the inputs.
    
    Parameters
    ----------
    par_mat : numpy.array of all parameter combinations in their original form. 

    Returns
    -------
    par_norm : standardnorm of par_mat
    scaler : scaler value for normalization 
    ''' 

    par_mat_ln = numpy.log(par_mat)     
    
    scaler = StandardScaler()
    par_norm = scaler.fit_transform(par_mat_ln)
    return par_norm, scaler



def par_transform(par_mat_new, scaler):
    '''
    par_transform : performs scaler transform on any new combination of parameter using the same scaler generated in par_fit_transform.

    Parameters 
    ----------
    par_mat_new : numpy.array of new parameter combinations in their original form.
    scaler : scaler value for normlization. 

    Returns
    -------
    par_norm_new : standardnorm of par_matnew
    '''
    par_mat_ln = numpy.log(par_mat_new)
    par_norm_new = scaler.transform(par_mat_ln)
    return par_norm_new


def plot_sim(y, dir,fname):
    '''
    plot_sim : plot's input vs arbitrary output

    Parameters 
    ----------
    y : numpy.array of y-axis values. Each row is new result.
    dir : directory where it saves the plot
    fname : filename of the saved plot 

    Returns
    -------
    Nothing
    '''
    num = y.shape[0]

    fig,ax = plt.subplots(num,1, figsize=[10,15])
    for i in range(num):
        ax[i,].plot(y[i,:])
    plt.xlabel('x (a.u.)')
    plt.ylabel('y (a.u.)')
    plt.show()
    fig.savefig(dir/fname)
plt.show()    

@tf.function()
# custom function for NN training
def log_mse(y_true, y_pred):
    return tf.math.log(tf.keras.losses.MSE(y_true, y_pred))

def network(x, y, training_folder, training_name):
    '''
    netwrok : NN model used for training surrogate model
           
    Parameters 
    ----------
    x : numpy.array of inputs to NN. Material parameters in their standard normalized form(par_norm). Axis = 0 should be equal to axis = 0 of y.
    y : numpy.array of y-axis values. Each row is new result.  Axis = should be equal to axis = 0 of x.
    training folder : folder where the training results will be stored after completion of training.
    training_name : name of hdf file in which the weights and biases will be stored

    Saves
    -------
    reg : regression model
    x_test : the x values used for testing
    y_test : the y values used for testing
    x_train : the x_values used for training and validation
    y_train : the y values used for training and validation
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, shuffle=False)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # Network Parameters
    max_filter = 256
    strides = [2,2,2,2]
    kernel = [8,8,8,8]
    map_size = 32

    nParams = X_train.shape[1]

    z_in = tf.keras.layers.Input(shape=(nParams,))
    z1 = tf.keras.layers.Dense(max_filter)(z_in)
    z1 = tf.keras.activations.swish(z1)
    z1 = tf.keras.layers.Dense(max_filter*map_size)(z1) #256 * 32
    z1 = tf.keras.activations.swish(z1)
    z1 = tf.keras.layers.Reshape((map_size,max_filter))(z1) # 32 by 256
    z2 = tf.keras.layers.Conv1DTranspose( max_filter//2, kernel[3], strides=strides[3], padding='SAME')(z1) # 64 by 128
    z2 = tf.keras.activations.swish(z2)
    z3 = tf.keras.layers.Conv1DTranspose(max_filter//4, kernel[2], strides=strides[2],padding='SAME')(z2) # 128 by 64
    z3 = tf.keras.activations.swish(z3)
    z4 = tf.keras.layers.Conv1DTranspose(max_filter//8, kernel[1], strides=strides[1],padding='SAME')(z3) # 256 by 32
    z4 = tf.keras.activations.swish(z4)
    z5 = tf.keras.layers.Conv1DTranspose(1, kernel[0], strides=strides[0],padding='SAME')(z4) # 512 by 1
    decoded_Y = tf.keras.activations.swish(z5)
    decoded_Y = tf.keras.layers.Reshape((Y_train.shape[1],))(decoded_Y)
    

    def scheduler(epoch, lr):
        if epoch <140:
             lr = 0.001
             return lr
        else :
            lr = lr * numpy.exp(-0.005)
            return lr
            
    checkpoint_path = training_folder / ("cp.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,  verbose=1)

        
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_log_mse', factor=0.50, patience=50, verbose=1, mode='min', min_delta=0.001, cooldown=0, min_lr=1e-5)
  

    log_folder = Path(training_folder)/training_name 
    tf_callbacks = [TensorBoard(log_dir=log_folder,
                         histogram_freq=1,
                         update_freq='epoch',
                         profile_batch=(2,10))]


    reg = Model(z_in,decoded_Y)
    reg.summary()
    

    es = EarlyStopping(monitor='val_loss', min_delta=1e-5, mode='min', verbose=1, patience=500, restore_best_weights=True)

    reg.compile(loss='mae',optimizer='adam', metrics=[log_mse])
    reg.fit(X_train,Y_train,shuffle=False, batch_size=192, epochs = 10000,
                validation_split=0.05,  callbacks=[es, tf_callbacks, reduce_lr])
    
    
    reg_name = Path(training_folder)/("%s_trained_model.h5" % training_name)
    reg.save(reg_name)
    return reg, reg_name, X_test, Y_test, X_train, Y_train
 
    
def main(sim_name, timestamp, exp, dataset):
    
    # create drectory
    dir, name = create_dir(sim_name, timestamp)
    print('DIRECTORY CREATED AT:', dir)

    train_data = h5.H5()
    train_name = dir/("%s_train_test.h5" % name)   #the file where all your training related data is getting stored
    train_data.filename = train_name.as_posix()
    train_data.root.dataset = str(dataset)
    print(dataset)
    train_data.save()
   
    
    # load datasets
    if exp == 'one': 
        y_norm, y_max, y_min, par_mat, ub, lb, n_var = load_dataset_one_exp(dataset, exp)
        print(y_norm.shape, y_max, y_min)
        train_data.root.y_max = y_max
        train_data.root.y_min = y_min
        train_data.root.ub = ub
        train_data.root.lb = lb
        train_data.root.n_var = n_var
    elif exp == 'two':
        y_norm, y_max, y_min, par_mat, ub, lb, n_var = load_dataset_one_exp(dataset, exp)
        train_data.root.y_max = y_max
        train_data.root.y_min = y_min
        train_data.root.ub = ub
        train_data.root.lb = lb
        train_data.root.n_var = n_var
    elif exp == 'both':
        y_norm, y1_max, y1_min, y2_max, y2_min, par_mat, ub, lb, n_var = load_dataset_both_exp(dataset, exp)
        train_data.root.y1_max = y1_max
        train_data.root.y1_min = y1_min
        train_data.root.y2_max = y2_max
        train_data.root.y2_min = y2_min
        train_data.root.ub = ub
        train_data.root.lb = lb
        train_data.root.n_var = n_var
    print('DATA LOADED FROM:', dataset)
    
    # select some random points and plot to see if you have loaded the correct datasets.
    num = 9
    idx = numpy.random.randint(0, y_norm.shape[0],num)
    plot_sim(y_norm[idx,:], dir, 'y_norm.png')

    # StandardScaler transform par_mat
    par_norm, scaler = par_fit_transform(par_mat)
    train_data.root.par_norm = par_norm
    scaler_name = dir/(timestamp + "_scaler.joblib")
    joblib.dump(scaler, scaler_name)  
    
    # train neural network
    reg, reg_name, x_test, y_test, x_train, y_train = network(par_norm, y_norm, dir, name)

    # save training data
    train_data.root.X_test = x_test
    train_data.root.Y_test = y_test
    train_data.root.X_train = x_train
    train_data.root.Y_train = y_train
    train_data.root.reg_name = str(reg_name)
    train_data.root.train_name = str(train_name)
    train_data.root.scaler_name = str(scaler_name)
    train_data.save()

    return dir, reg_name, train_name, scaler_name

if __name__ == "__main__":
    main(sim_name, timestamp, exp, dataset)
