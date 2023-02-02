import NN_training
import bayes
from pathlib import Path
import datetime

nn_trained = 'yes'

dataset = Path("./Dataset/one_diode_4_temp_50K.h5")
exp_path = Path("./Experimental_data/one_diode_4_temp_50K_21768.h5")
sim_name = 'one_diode_4_temp_50K_21768'
exp = 'one'
walkers = 200
sigma = 1e-4
sp = 1 # find strating points from traiing dataset ; 2 = by finding best point using pymoo and high probability relocation, = 3 a cmobinaton of 1 and 2
    
if nn_trained == 'no':
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path, reg_path,train_path, scaler_path = NN_training.main(sim_name, timestamp, exp, dataset)
    print('reg_path', reg_path, train_path, scaler_path)
    print('train_path', train_path)
    print('scaler_path', scaler_path)
elif nn_trained == 'yes':
    dir_path = Path('./Results/20230127-110643_one_diode_4_temp_high_start_42970')
    reg_path = dir_path/('20230127-110643_one_diode_4_temp_high_start_42970_trained_model.h5')
    train_path = dir_path/('20230127-110643_one_diode_4_temp_high_start_42970_train_test.h5')
    scaler_path = dir_path/('20230127-110643scaler.joblib')
    
bayes.main(dir_path, sim_name, reg_path, train_path, scaler_path, exp_path, exp, walkers, sigma, sp)
