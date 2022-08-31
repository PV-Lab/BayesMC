import numpy
import sys
import h5
import os
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.factory import get_termination, get_reference_directions, get_algorithm
from pymoo.factory import get_sampling
from pymoo.util.display import SingleObjectiveDisplay
import numpy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from scipy import interpolate
import addict
import emcee
import de
import de_snooker
import emcee.state
import scipy.spatial
from sklearn.cluster import KMeans
import emcee.autocorr as autocorr
import pandas
import datetime
import corner
from NN_training import log_mse

def new_range(flat_chain):
    '''
    new_range : to set range of corner plot

    Parameters
    ----------
    flat_chain : flat chain for plotting

    Returns
    -------
    new range for lower and upper bound
    '''
    lb_data, mid_data, ub_data = numpy.percentile(flat_chain, [5, 50, 95], 0)

    distance = numpy.max(
        numpy.abs(numpy.array([lb_data - mid_data, ub_data - mid_data])), axis=0
    )

    lb_range = mid_data - 2 * distance
    ub_range = mid_data + 2 * distance

    return numpy.array([lb_range, ub_range])

def run_walkers(auto_state,sampler,data,cur_path):
    '''
    run_walkers: final mcmc run
    '''
    chain = None
    prob = None
    taus = None
    tau_check = 100
    finished = False
    step = 0
    tol = 52

    state = auto_state
    
    
    while not finished:
        state = next(sampler.sample(state, iterations=1))

        p = state.coords
        ln_prob = state.log_prob

        chain = addChain(chain, p[:,numpy.newaxis,:])
        mother_chain = chain
        prob = addChain(prob, ln_prob[:,numpy.newaxis])
        mother_prob = prob
        #print(f"step {step}")

        if step and step % 1000 == 0:
            
            plot_mixing(chain,cur_path)                  # mixing graphs to check if the variables are converging or not
            burn = int(numpy.ceil(numpy.max(tau)) * 2)
           
            burn_chain = chain[:,:burn,:]
            burn_prob = prob[:,:burn]

            chain = chain[:,burn:,:]
            prob = prob[:,burn:]
            #plot_check(state, sampler)          # to plot jv curves from the current position of all walkeres overlayed on top of the jv exp

        if step % tau_check == 0:
            try:
                tau = sampler.get_autocorr_time(tol=tol)
            
                print(f"step {step}  progress {step/(tau*tol)}")

                if not numpy.any(numpy.isnan(tau)):
                    finished = True

            except autocorr.AutocorrError as err:
                tau = err.tau
            
                print(f"step {step}  progress {step/(tau*tol)}")

            taus = addChain(taus, tau[:,numpy.newaxis])
        step = step + 1

    print("done processing")
    
    burn = int(numpy.ceil(numpy.max(tau)) * 2)

    burn_chain = chain[:,:burn,:]
    burn_prob = prob[:,:burn]

    chain = chain[:,burn:,:]
    prob = prob[:,burn:]

    print("finished")

    #data = h5.H5()
    #data.filename = f"bayes_save.h5"
    data.root.burn_chain = burn_chain
    data.root.burn_prob = burn_prob
    data.root.chain = chain
    data.root.prob = prob
    data.root.tau = tau
    data.root.motherchain = mother_chain
    data.root.motherprob = mother_prob
    data.save()
    return chain, prob

cm_plot = plt.cm.gist_rainbow

def get_color(idx, max_colors, cmap):
    return cmap(1.0 * float(idx) / max_colors)

def flatten(chain):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
    return flat_chain

def plot_mixing(chain,cur_path):
    f_mixing = Path(cur_path)/("mixing")
    if os.path.exists(f_mixing):
        print("file already exists")
    else:
        os.mkdir(f_mixing)
    chain_length = chain.shape[1]
    x = numpy.linspace(0, chain_length - 1, chain_length)
    for i in range(chain.shape[2]):
        plt.figure(figsize=[15,7])
        
        for j in range(chain.shape[0]):
            plt.plot(x, chain[j, :, i], color=get_color(j, chain.shape[0] - 1, cm_plot))
        fig_path = Path(f_mixing)/("mixing_%i.png" %i)
        plt.savefig(fig_path)
        plt.close('all')
        #plt.savefig(Path(f_mixing)/("mixing_{i}.png"))

    flat_chain = flatten(chain)

    fig = corner.corner(
        flat_chain,
        quantiles=(0.05, 0.5, 0.95),
        show_titles=True,
        bins=20,
        range=new_range(flat_chain).T,
        use_math_text=True,
        title_fmt=".2g",
    )
    plt.savefig(Path(f_mixing)/("corner.png"))
    plt.close('all')


def plot_start(auto_state, y_exp_norm, reg, scaler, lb, ub,cur_path, data):
    '''
    plot_stat : plot the predicted Y for all the starting points super imposed on the normalized experimental data

    Parameters: 
    auto_state : coords of all the walkers
    y_exp_norm : numpy array of normalized experimental data
    reg : regression model
    scaler : scaler to tansform the parameteres
    lb: lower bound
    ub : upper bound
    cur_path : where the figures will be stores
    '''
    f_high_start = Path(cur_path)/("high_start")
    if os.path.exists(f_high_start):
        print("file already exists")
    else:
        os.mkdir(f_high_start)

    coords = auto_state.coords
    #coords = auto_state
    lb = numpy.log(lb)
    ub = numpy.log(ub)
    theta_trans = numpy.array(coords) * (ub-lb) + lb
    theta_actual = numpy.exp(theta_trans)
    
    theta_norm = theta_transform(theta_trans, scaler)
    jv_predicted_from_theta = reg.predict(theta_norm)

    data.root.walkers_start_actual = theta_actual
    data.root.jv_predict = jv_predicted_from_theta
    data.save()

    
    for i in range(jv_predicted_from_theta.shape[0]):
        plt.figure(figsize=[10,10])
        plt.plot(jv_predicted_from_theta[i,:], label="nn")
        plt.plot(numpy.squeeze(y_exp_norm), label="target")
        plt.title(f"{theta_actual[i,:]}")
        plt.legend()
        fig_path = Path(f_high_start)/("prob_%i.png" %i)
        plt.savefig(fig_path)
        plt.close('all')
    
    



def auto_high_probability(sampler, start, iterations=100):
    '''
    auto_high_probability : relocates all the walkers to high probability region

    Parameters 
    ----------
    sampler : emcee sampler model
    start : starting point for all the walkers
    interation : number of generations to run the emcee sampler before stopping

    Returns
    -------
    state : high probability starting point for all the walkers
    auto_chain : chain from the mcmc run
    auto_probability : probability of all the positions in the chain

    '''
    auto_chain = None
    auto_probability = None
    
    state = emcee.state.State(start)
    
    log_prob = None
    rstart = None
    
    finished = None
    prev_prob = -1e308

    while not finished:
        chain, probability = auto_high_probability_iterations(sampler, iterations, state)

        # store chain
        auto_chain = addChain(auto_chain, chain)
        auto_probability = addChain(auto_probability, probability)

        best_chain, best_prob = select_best_kmeans(auto_chain, auto_probability)

        #print(f"best_chain {best_chain}")
        #print(f"best_prob {best_prob}")
        
        state.coords = best_chain
        state.log_prob = best_prob
        state.random_state = None
        
        best_prob = numpy.max(best_prob)
        
        change = numpy.abs(best_prob - prev_prob)/numpy.abs(prev_prob)
        
        if change < 0.001:
            finished=True
        else:
            print(f"Auto high probability has not converged yet, change {change} > 0.001")
            prev_prob = best_prob
        
    return state, auto_chain, auto_probability

def auto_high_probability_iterations(sampler, iterations, state):

    auto_chain = None
    auto_probability = None
    best = -numpy.inf

    for i in range(iterations):
        state = next(
            sampler.sample(
                state,
                iterations=1,
            )
        )

        p = state.coords
        ln_prob = state.log_prob
        random_state = state.random_state

        if any(ln_prob > best):
            best = numpy.max(ln_prob)

        accept = numpy.mean(sampler.acceptance_fraction)

        auto_chain = addChain(auto_chain, p[:, numpy.newaxis, :])
        auto_probability = addChain(auto_probability, ln_prob[:, numpy.newaxis])

        if i%10 == 0:
            print(f"auto run: idx: {i} accept: {accept:.3f} max ln(prob): {best:.5f}")

    sampler.reset()
    return auto_chain, auto_probability

def select_best_kmeans(chain, probability):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
    flat_probability = numpy.squeeze(probability.reshape(-1, 1))

    # unique
    flat_chain_unique, unique_indexes = numpy.unique(
        flat_chain, return_index=True, axis=0
    )
    flat_probability_unique = flat_probability[unique_indexes]

    # remove low probability
    flat_prob = numpy.exp(flat_probability_unique)
    max_prob = numpy.max(flat_prob)
    min_prob = max_prob / 10  # 10% of max prob cutoff

    selected = (flat_prob >= min_prob) & (flat_prob <= max_prob)

    flat_chain = flat_chain_unique[selected]
    flat_probability = flat_probability_unique[selected]

    if len(flat_chain) > (2 * chain_shape[0]):
        # kmeans clustering
        km = KMeans(chain_shape[0])
        km.fit(flat_chain)

        dist = scipy.spatial.distance.cdist(flat_chain, km.cluster_centers_)

        idx_closest = numpy.argmin(dist, 0)

        closest = dist[idx_closest, range(chain_shape[0])]

        best_chain = flat_chain[idx_closest]
        best_prob = flat_probability[idx_closest]
    else:
        pop_size = chain.shape[0]
        sort_idx = numpy.argsort(flat_probability_unique)
        sort_idx = sort_idx[numpy.isfinite(sort_idx)]

        best = sort_idx[-pop_size:]

        best_chain = flat_chain_unique[best, :]
        best_prob = flat_probability_unique[best]

    return best_chain, best_prob


def addChain(*args):
    temp = [arg for arg in args if arg is not None]
    if len(temp) > 1:
        return numpy.concatenate(temp, axis=1)
    else:
        return numpy.array(temp[0])



def montecarlo(best_scaled, log_probability_vec, y_exp_norm,sigma,reg, scaler, lb, ub ):
    '''
    montecarlo : mcmc sampler

    Parameters 
    ----------
    best_scaled : parameters predicted res.X in pymoo space, which is from 0 to 1.
    log_proability_vec : ln transforms the theta from pymoo space. then performs scaler transform on the ln transformed theta.
                         calculated the predicted y curve using trained surrogate model
    y_exp_norm : normalized experimental data
    sigma : gaussian error factor
    reg : regression model
    scaler : scaler for parameter transform
    lb : upper bound
    ub : lower bound
    
    Returns
    -------
    sampler : emcee sampler model
    start : starting point for the walkers
    '''


    chain = None
    prob = None
    taus = None

    tau_check = 100
    lb = numpy.log(lb)
    ub = numpy.log(ub)

    nwalkers = 512
    print(best_scaled.shape)
    ndim = len(best_scaled)
    print(nwalkers, ndim)

    gamma0 = 2.38/ numpy.sqrt(2 * ndim)

    start = numpy.array(best_scaled) * numpy.random.normal(1.0, 1e-2, (nwalkers, ndim))
    start_random = numpy.random.rand((nwalkers, ndim))
    start = numpy.clip(start, 0, 1)
    start_random = numpy.clip(start_random, 0, 1)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_vec, vectorize=True, 
                                    args=(y_exp_norm, sigma,reg, scaler, lb, ub),
                                    moves=[(de_snooker.DESnookerMove(), 0.1),
                                        (de.DEMove(), 0.9*0.9),
                                        (de.DEMove(gamma0), 0.9*0.1),])
    return sampler, start, start_random


def plot_best(best_norm, y_exp_norm, reg, sub_path):
    '''
    plot_best : plots the predicted y at the best point

    Parameters 
    ----------
    best_norm : the scaler normalized value of the best point predicted by pymoo
    y_exp_norm : normalized experimental data
    reg : regression model
    dir : directory where the plot will be saved

    Returns
    -------
    Nothing
    '''
    y_predict = reg.predict(best_norm)
    
    err = numpy.sum( (numpy.squeeze(y_predict) - numpy.squeeze(y_exp_norm))**2)
    
    plt.figure(figsize=[10,10])
    plt.plot(numpy.squeeze(y_predict), label="nn")
    plt.plot(numpy.squeeze(y_exp_norm), label="target")
    plt.title(f"{best_norm} err {err}")
    plt.legend()
    plt.savefig(Path(sub_path)/('best_point_predict'))
    plt.show()
    plt.close('all')
    

def save_best(train_path, best_scaled, best_ln):
    '''
    save_best: saves the best point found by pymoo

    Parameters 
    ----------
    train_path : location of the file where this data is stored
    best_sclaed : the best point in the pymoo space
    best_ln : the best point in natural log sapce

    Returns
    -------
    Nothing

    '''

    td = h5.H5()
    td.filename = train_path
    td.load()
    td.root.best_scaled = best_scaled
    td.root.best_point_ln = best_ln
    td.root.best_point = numpy.exp(best_ln)
    td.save()

def plot_sim(y, y_predict, dir, fname):
    '''
    plot_sim : plot's y_norm and predited y vs arbitrary output

    Parameters 
    ----------
    y : numpy.array of y-axis values. Each row is new result.
    y_predict : numpy array of y-axis of predicted output
    dir : the directory where the plot is stores
    fname : the filename of the sved plot

    Returns
    -------
    Nothing
    '''
    num = y.shape[0]

    fig,ax = plt.subplots(num,1, figsize=[10,15])
    for i in range(num):
        ax[i,].plot(y[i,:])
        ax[i,].plot(y_predict[i,:],'--')
    plt.xlabel('x (a.u.)')
    plt.ylabel('y (a.u.)')
    plt.show()
    fig.savefig(dir/fname)
    plt.close('all')
    

def check_nn(reg, train_path, dir, fname):

    td = h5.H5()
    td.filename = train_path
    td.load()
    x_test = td.root.X_test
    y_test = td.root.Y_test

    idx = numpy.random.randint(0, x_test.shape[0],5)
    y_predict = reg(x_test[idx,:])
    plot_sim(y_test[idx,:], y_predict, dir, fname)




def load_training_data(train_path, exp):
    '''
    load_training_data : loads training data from hdf5

    Parameters 
    ----------
    train_path : file path to training data file
    exp : identifies which experiment we are using

    returns
    -------
    ub
    lb
    n_var
    y1_max
    y2_min
    y2_max
    y2_min

    '''
    td = h5.H5()
    td.filename = train_path
    td.load()
    ub = td.root.ub
    lb = td.root.lb
    n_var = td.root.n_var
    if exp == 'both':
        y1_max = td.root.y1_max
        y1_min = td.root.y1_min
        y2_max = td.root.y2_max
        y2_min = td.root.y2_min
        return ub, lb, n_var, y1_max, y1_min, y2_max, y2_min

    elif exp == 'one' or exp =='two':
        y_max = td.root.y_max
        y_min = td.root.y_min
        return ub, lb, n_var, y_max, y_min

    



def load_exp_data(exp_path, exp):
    '''
    load_exp_data : loads interpolated experimental data from hdf5

    Parameters 
    ----------
    exp_path : file path to experimental file
    exp : identifies which experiment we are using

    returns
    -------
    y_exp_1 : y-axis of 1 or multiple illumination intensities of experiment 1
    y_exp_2 : y-axis of 1 or multiple illumination intensities of experiment 2

    '''

    exp_data = h5.H5()
    exp_data.filename = exp_path.as_posix()
    exp_data.load()
    if exp =='both':
        y_exp_1 = exp_data.root.y_exp_1
        y_exp_2 = exp_data.root.y_exp_2
        return y_exp_1, y_exp_2
    elif exp =='one':
        y_exp = exp_data.root.y_exp_1
        print(y_exp.shape)
        return y_exp
    elif exp =='two':
        y_exp = exp_data.root.y_exp_2
        return y_exp


def y_exp_transform(y_exp, max, min):
    '''
    y_exp_transform : norm of the experimental data using the min max from the simulated dataset

    Parameters 
    ----------
    y : experimental data
    max : max of the simulated data
    min : min of the simulated data

    returns
    -------
    y_exp_norm : norm of the experimetnal data

    '''
    y_exp_norm = (y_exp-min)/(max-min)

    return y_exp_norm


def log_norm_pdf_vec(y,mu,sigma):
    
    '''
    log_norm_pdf_vec : calculates the gaussian error between the predicted jv and the experimental data

    Parameters 
    ----------
    y : jv predicted from theta
    mu : normalized experimental data
    sigma : gaussian error parameter

    returns
    -------
    error : error between the jv predicted from theta and experimental data

    '''
    error = -0.5*numpy.sum((((y-mu)**2/(sigma**2)) + numpy.log(2*3.14*(sigma**2))), axis = 1)
    return error


def log_probability_vec(theta,y_exp_norm,sigma, reg, scaler, lb, ub):

    '''
    log_probability_vec : ln transforms the theta from pymoo space. then performs scaler transform on the ln transformed theta.
                          calculated the predicted jv curve using trained surrogate model
   
    Parameters 
    ----------
    theta : numpy array of the parameters predicted by pymoo. theta is pymoo space (0 to 1)
    y_exp_norm : minmax transformed experimental data
    sigma : gaussian error parameter
    reg : regression model
    scaler : scaler value for normalization
    lb : lower bound
    ub : upper bound

    returns
    -------
    errors : error between the jv predicted from theta and experimental data

    '''

    theta_trans = numpy.array(theta) * (ub-lb) + lb    
    theta_norm = theta_transform(theta_trans, scaler)
    jv_predicted_from_theta = reg.predict(theta_norm)
    errors = log_norm_pdf_vec(jv_predicted_from_theta, y_exp_norm, sigma)
    return errors



def theta_transform(theta, scaler):
    
    '''
    theta_transform : does transform of theta predicted by pymoo

    Parameters 
    ----------
    theta : numpy array of the parameters predicted by pymoo. theta is pymoo space (0 to 1)
    scaler : scaler value for normalization
    
    returns
    -------
    theta_norm : scaler transformed theta  
    
    '''

    theta_norm = scaler.transform(theta)
    return theta_norm



def pymoo(lb, ub, n_var, y_exp_norm, reg, scaler):
    
    '''
    pymoo : optimizer to find a good point
           
    Parameters 
    ----------
    lb : lower bound in orignal form. NOT ln transformed
    ub : upper bound in original form. NOT ln transformed
    n_nar : number of variables
    y_exp_norm : minmax transformed experimental data
    reg : regression model
    data : 
    scaler : 
    res.X, numpy.exp(best), best_norm, sigma


    Returns
    -------
    res.X : parameters predicted res.X in pymoo space, which is from 0 to 1. 
    numpy.exp(best) : predicted parameters in original space. 'best' is the parameters in ln space. best =  res.X * ln(par_mat)
    best_norm : predicted parameters in min max scaler transformed space
    sigma : gaussian error parameter
    
    '''

    lb = numpy.log(lb)
    ub = numpy.log(ub)

    termination = get_termination("n_gen", 1000)
    
    sigma = 1e-3
    popsize = 200
    
    class MyProblem(Problem):

        def __init__(self, jv_exp_norm, sigma, reg, lb, ub ):
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=numpy.zeros(lb.shape), xu=numpy.ones(ub.shape), elementwise_evaluation=False)
            #self.scaler = joblib.load("par_mat.joblib")
            self.scaler = scaler
            self.jv_exp_norm = jv_exp_norm
            self.sigma = sigma
            self.reg = reg
            self.real_lb = lb
            self.real_ub = ub
        def _evaluate(self, theta, out, *args, **kwargs):            
            #print(f"shape {theta.shape}")
            error = -log_probability_vec(theta, self.jv_exp_norm, self.sigma, self.reg, self.scaler, self.real_lb, self.real_ub)
            out["F"] = error

    
    algorithm = get_algorithm("cmaes", popsize=popsize, display=SingleObjectiveDisplay())

    #ref_dirs = get_reference_directions("energy", 1, popsize, seed=1)
    #algorithm = get_algorithm("unsga3", ref_dirs=ref_dirs, pop_size=popsize )

    problem = MyProblem(y_exp_norm, sigma, reg, lb, ub)

    res = minimize(problem, algorithm, verbose=True)
    
    best_ln = (res.X * (ub-lb) + lb).reshape(1,-1)                     #ln space
    best_norm = theta_transform(best_ln, problem.scaler)               #scaler transformed space
    print(f"Best point at {best_ln} with score  {res.F}")
    return res.X, best_ln, best_norm, sigma

def plot_corner(sub_path, chain, ub_ln, lb_ln):
    flat_chain = flatten(chain)
    flat_chain_trans = (flat_chain * (ub_ln-lb_ln) + lb_ln)
    flat_chain_actual = numpy.exp(flat_chain_trans)
    flat_chain_actual[:,-1] = numpy.log10(flat_chain_actual[:,-1])
    flat_chain_actual[:,0] = flat_chain_actual[:,0] * 1e3
    label = [r'$R_\mathrm{s} [\Omega \mathrm{cm^2}]$', r'$R_\mathrm{sh} [\mathrm{k}\Omega \mathrm{cm^2}]$' ,r'$n_\mathrm{id}$' , r'$\log(J_{0} \mathrm{[mA /cm^2 ]})$']
    fig = corner.corner(
        flat_chain_actual,
        labels = label,
        label_kwargs={"fontsize": 20},
        quantiles=(0.05, 0.5, 0.95),
        show_titles=True,
        title_kwargs={"fontsize": 10},
        bins=30,
        plot_contours = True,
        range=new_range(flat_chain_actual).T,
        color = "#00316E",
        use_math_text=True,
        max_n_ticks = 4,
        title_fmt=".3f"
    )
    fig.subplots_adjust(right=1.6,top=1.6)
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=20)
    plt.savefig(sub_path/("corner.pdf"),dpi=300,pad_inches=0.2,bbox_inches='tight')
plt.show()

def main(dir_path, name, reg_path, train_path, scaler_path,exp_path, exp):
    identity = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sub_dir = identity + name + '_bayes'
    print(sub_dir)
    sub_path = dir_path/("%s" % sub_dir)
    os.mkdir(sub_path)
    
    results = h5.H5()
    results_name = sub_path/("%s.h5" % sub_dir)
    results.filename = results_name.as_posix() #the file where all your training related data is getting stored
    results.save()
    print(results.filename)
    
    # load trained regression model
    reg = tf.keras.models.load_model(reg_path, custom_objects={'log_mse':log_mse})

    # load upper bound, lower bound and min max of different experiments
    if exp == 'both':
        ub, lb, n_var, y1_max, y1_min, y2_max, y2_min = load_training_data(train_path, exp)
    elif exp == 'one' or exp =='two':
        ub, lb, n_var, y_max, y_min = load_training_data(train_path, exp)

    # load scaler
    scaler = joblib.load(scaler_path)

    
    # check NN training
    fname = 'nn_check_plot'
    check_nn(reg, train_path, sub_path, fname)


    # load experimental data 
    '''
    Before this step is done make sure you experimental data is already interpolated to the correct size.
    '''
    if exp == 'both':
        y_exp_1 , y_exp_2 = load_exp_data(exp_path, exp)
        plt.plot(y_exp_1)
        plt.show()
        plt.plot(y_exp_2)
        plt.show()
        results.root.y_exp_1 = y_exp_1
        results.root.y_exp_2 = y_exp_2
    elif exp == 'one':
        y_exp = load_exp_data(exp_path, exp)
        plt.plot(y_exp)
        plt.show()
        results.root.y_exp = y_exp
    elif exp == 'two':
        y_exp = load_exp_data(exp_path, exp)
        plt.plot(y_exp)
        plt.show()
        results.root.y_exp = y_exp
    results.save()
    # transform experimental data
    if exp == 'both':
        y_exp_1_norm = y_exp_transform(y_exp_1, y1_max, y1_min)
        y_exp_2_norm = y_exp_transform(y_exp_2, y2_max, y2_min)
        y_exp_norm = numpy.concatenate((y_exp_1_norm, y_exp_2_norm))
        plt.plot(y_exp_norm)
    elif exp == 'one' or exp == 'two':
        y_exp_norm = y_exp_transform(y_exp, y_max, y_min)
        plt.plot(y_exp_norm)
    results.root.y_exp_norm = y_exp_norm
    results.save()
    # run optimizer to find starting point
    best_scaled, best_ln, best_norm, sigma = pymoo(lb, ub, n_var, y_exp_norm, reg, scaler)
    best_actual = numpy.exp(best_ln)
    print("best_point:", best_actual)
    results.root.best.best_scaled = best_scaled
    results.root.best.best_ln = best_ln
    results.root.best.best_actual = best_actual
    results.save()

    # plot the y_predict at the best point and compare with experimental result
    plot_best(best_norm, y_exp_norm, reg, sub_path)

    # run auto high probility relocation step
    sampler, start, start_random = montecarlo(best_scaled, log_probability_vec, y_exp_norm, sigma, reg, scaler, lb, ub)
    print(start.shape)
    auto_state, auto_chain, auto_probability = auto_high_probability(sampler, start)
    results.root.walkers_start = auto_state.coords
    results.save()
    #auto_state = emcee.state.State(start_random)

    # plot the predicted y on top of the experimental data for the starting point of each walker
    plot_start(auto_state, y_exp_norm, reg, scaler,lb, ub, sub_path, results)
    

    # run final bayes step
    chain, prob = run_walkers(auto_state,sampler, results, sub_path)


    
    ub_ln = numpy.log(ub)
    lb_ln = numpy.log(lb)
    
    plot_corner(sub_path, chain, ub_ln,lb_ln)
    

if __name__ == "__main__":
    main(dir_path, name, reg_path, train_path, scaler_path,exp_path, exp)







    


