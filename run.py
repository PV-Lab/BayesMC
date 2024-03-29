import NN_training
import bayes
from pathlib import Path
import datetime

nn_trained = 'no'

dataset = Path("./Dataset/one_diode_4_temp_30000.h5")
exp_path = Path("./Experimental_data/one_diode_4_temp_781.h5")
sim_name = 'one_diode'
exp = 'one'
if nn_trained == 'no':
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path, reg_path,train_path, scaler_path = NN_training.main(sim_name, timestamp, exp, dataset)
    print('reg_path', reg_path, train_path, scaler_path)
    print('train_path', train_path)
    print('scaler_path', scaler_path)
elif nn_trained == 'yes':
    dir_path = Path('./Results/20220826-040829one_diode')
    reg_path = dir_path/('20220826-040829one_diode_trained_model.h5')
    train_path = dir_path/('20220826-040829one_diode_train_test.h5')
    scaler_path = dir_path/('20220826-040829scaler.joblib')
    
bayes.main(dir_path, sim_name, reg_path, train_path, scaler_path, exp_path, exp)
