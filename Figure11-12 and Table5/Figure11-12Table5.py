#import driver
from tf_agents.drivers import py_driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from sklearn.metrics import r2_score

#import environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

#import replay buffer
from tf_agents import replay_buffers as rb

#import agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.utils import value_ops
from tf_agents.trajectories import StepType


#other used packages
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as ax
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_agents
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
#from custom_actor_distribution_network import ActorDistributionNetwork
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
#from tf_agents.networks.normal_projection_network import NormalProjectionNetwork
from custom_normal_projection_network import NormalProjectionNetwork

import os,gc
from sklearn.model_selection import train_test_split

import random
# TensorFlow and tf.keras
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
print(tf.__version__)
import os
import gc
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import TimeStep
from tf_agents.trajectories import StepType

#To limit TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
tf_agents.system.multiprocessing.enable_interactive_mode()
data_path = "D:/Large_Files_For_Learning/Project_Result_Data/with_YZhang/VIP-DFNN/Data/"
gc.collect()



#load the simulated data
den_inj = np.array(pd.read_csv("inputs802.csv")) #[r, h]
param10 = np.array(pd.read_csv("output10.csv"))  #[y, h]

#scale the density by 1,000,000
den_inj[:,0:den_inj.shape[1]-2] = np.array(den_inj)[:,0:den_inj.shape[1]-2]*1000000

#Set the sample size
sample_size = 200000

#Check for duplicates
print('param10 shape before dropping duplicates',param10.shape, 
      'after duplicates are dropped', pd.DataFrame(param10).drop_duplicates().shape)
print('den_inj shape before dropping duplicates',den_inj.shape, 
      'after duplicates are dropped', pd.DataFrame(den_inj).drop_duplicates().shape)

#Set the smaple size for training, validation, and testing
tr_size = int(sample_size*0.8)
val_size = int(sample_size*0.1)
te_size = int(sample_size*0.1)
train_val_size = val_size+tr_size

#Get the train, validation, and test data
train_x, val_x, train_y, val_y = train_test_split(den_inj[0:train_val_size], param10[0:train_val_size,0:8], test_size=val_size/(tr_size+val_size), random_state=48)
test_x = den_inj[-te_size:]  #x:[r,h]
test_y = param10[-te_size:,0:8]  #y:the 8 adsorption parameters

train_r = train_x[:,0:-2] #r:density curves
val_r = val_x[:,0:-2]
test_r = test_x[:,0:-2]

train_h = train_x[:,-2:]  #h:The two injection components
val_h = val_x[:,-2:]
test_h = test_x[:,-2:]

train_yh = np.append(train_y,train_h,axis=1)  #Inputs to fw_FNN
val_yh = np.append(val_y,val_h,axis=1)
test_yh = np.append(test_y,test_h,axis=1)


#Calculate sample mean and sd for later normalization
train_xm = np.mean(train_x,axis=0)
train_ym = np.mean(train_y,axis=0)
train_rm = np.mean(train_r,axis=0)
train_hm = np.mean(train_h,axis=0)
train_yhm = np.mean(train_yh,axis=0)

train_xsd = np.std(train_x,axis=0)
train_ysd = np.std(train_y,axis=0)
train_rsd = np.std(train_r,axis=0)
train_hsd = np.std(train_h,axis=0)
train_yhsd = np.std(train_yh,axis=0)

#Combine the training and validation data
train_val_yh = np.append(train_yh,val_yh,axis=0)
train_val_yhm = np.mean(train_val_yh,axis=0)
train_val_yhsd = np.std(train_val_yh,axis=0)

train_val_r = np.append(train_r,val_r,axis=0)
train_val_rm = np.mean(train_val_r,axis=0)
train_val_rsd = np.std(train_val_r,axis=0)

train_val_y = np.append(train_y,val_y,axis=0)
train_val_ym = np.mean(train_val_y,axis=0)
train_val_ysd = np.std(train_val_y,axis=0)

train_val_x = np.append(train_x,val_x,axis=0)
train_val_xm = np.mean(train_val_x,axis=0)
train_val_xsd = np.std(train_val_x,axis=0)

param_mean = np.mean(train_yh,axis=0)
param_std = np.std(train_yh,axis=0)


#Define the metric for model that predict density curves 
def report_metric(pred_y, true_y):
    """
    This function computes the metric M:=1 - avg(SSE/Energy),
    where SSE is the sum of squared prediction errors of a sample,
    Energy is the sum of squares of y entries for a sample,
    and the average avg() is taken over the samples.
    
    Inputs.
    --------
    true_y:np.ndarray, the true y values, each row is a sample
    pred_y:np.ndarray, the predictions from the model
    
    Outputs.
    M1:float, the metric which lies between [0,1], with 1 implies fit perfectness.
    """
    #Performance with sum of squares as a percentage of the Energy
    SSE = np.sum((pred_y-true_y)**2,axis=1) #sum of squares of error for each sample (row)
    Energy = np.sum((true_y)**2,axis=1) #density energy of each sample
    ratio = SSE/Energy
    MSE = np.mean(ratio) #Mean of SSE/(density energy)
    M1 = 1-MSE
        
    return M1
    


#Define the function for training a neural net model
def build_fwmodel(loss_func = 'L1', #Loss function to be used, "L1L2":error is L1 norm, bias and weights are L2 norms
                activ = 'tanh', #The activation function
                hid_layers = (25,36), #(1st hidden layer nodes, 2nd hidden layer nodes)
                bias_regu_cosnt = 0.01, #The regularization coeff. for bias terms
                w_regu_const = 0.01, #The regularization coeff. for weights
                show_process = 0, #If equals 1, show the training process, if equals 0, do not show. 
                output_size = 800
                 ):
    """
    This function returns an un-trained fw-FNN.
    """   
    #Build the model structure
    model = tf.keras.Sequential()
    if loss_func == 'L1': 
        error_loss = 'MAE'
        for node_num in hid_layers:
            model.add(tf.keras.layers.Dense(node_num, activation=activ, 
                                     kernel_initializer='glorot_uniform', 
                                     bias_initializer='glorot_uniform',
                                     kernel_regularizer=tf.keras.regularizers.l1(w_regu_const),
                                     bias_regularizer=tf.keras.regularizers.l1(bias_regu_cosnt))) 
    elif loss_func == 'L2':
        error_loss = 'MSE'
        for node_num in hid_layers:
            model.add(tf.keras.layers.Dense(node_num, activation=activ, 
                                     kernel_initializer='glorot_uniform', 
                                     bias_initializer='glorot_uniform',
                                     kernel_regularizer=tf.keras.regularizers.l2(w_regu_const),
                                     bias_regularizer=tf.keras.regularizers.l2(bias_regu_cosnt)))
             
    model.add(tf.keras.layers.Dense(output_size,activation='relu'))   
    model.compile(optimizer='adam', loss=error_loss, metrics=['MSE'])
    return model

#Build fw-FNN (with the optimal hyper-parameters) that maps from 8 adsorption parameters to density
#Only the train-val dataset is used for training this fw-FNN
tf.random.set_seed(0)
###################--fw-FNN
fw_FNN = build_fwmodel(loss_func = 'L2', #Loss function to be used, "L1L2":error is L1 norm, bias and weights are L2 norms
                       activ='tanh', #The activation function
                       hid_layers= (50,100,100,100), #(1st hidden layer nodes, 2nd hidden layer nodes)
                       bias_regu_cosnt= 0.001, #The regularization coeff. for bias terms
                       w_regu_const= 0.01, #The regularization coeff. for weights
                       output_size = 800
         )


early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20) 
history = fw_FNN.fit((train_val_yh-train_val_yhm)/train_val_yhsd, train_val_r, epochs=1500, 
                      batch_size=10000, shuffle=True, verbose=2, callbacks=[early_stop])


#fw-FNN Performance
train_pred = fw_FNN.predict((train_val_yh-train_val_yhm)/train_val_yhsd)
train_met = report_metric(train_pred, train_val_r)      
test_pred = fw_FNN.predict((test_yh-train_val_yhm)/train_val_yhsd)
test_met = report_metric(test_pred, test_r)                                     
print('fw-FNN:', 
      "Train_metric:",round(train_met,3),
      "Test_metric:",round(test_met,3),
      "Train_MSE:", round(history.history['MSE'][-1],3),
     ) 
fw_layers = [l for l in fw_FNN.layers]
del(train_x, val_x, train_y, val_y, test_x, test_y,
    train_r,val_r,test_r,train_h,val_h,test_h,train_yh,
    val_yh,test_yh,train_xm,den_inj,param10,train_val_r,
    train_val_x,train_val_y,train_val_yh,fw_FNN)
gc.collect()


#Create Fwmodel(x0)=y0
def Fwmodel(input_val
            #train_param_mean=np.array([50.1139665 , 49.98913638, 49.98965779, 49.93903001, 50.00064762, 50.02645835, 49.8903621 , 50.11872005, 14.97070843, 14.97756978]),
            #train_param_std=np.array([28.84737586, 28.86371378, 28.91718066, 28.86282312, 28.8475052, 28.90033581, 28.80407931, 28.87913465,  8.66257618,  8.67392796])
           ):
    #It uses the global variable fw_layers.
    x = np.reshape(input_val,[1,-1])
    #output_val = (x-train_param_mean)/train_param_std
    output_val = x
    
    for i in range(len(fw_layers)):                     
        output_val = fw_layers[i](output_val)
    return output_val[0]
    #output_val = fw_FNN((x-param_mean)/param_std)[0].numpy()



#Create Fwmodel(x0) = y0
np.random.seed(3)
xe = np.append(np.random.randint(low=40, high=60, size=8),
               np.random.randint(low=0, high=30, size=2)
              )
print('xe before transform =',xe)
xe = (xe-param_mean)/param_std  #transform xe
state_dim = len(xe)-2
ye = Fwmodel(input_val=xe)
#ye = Fwmodel(input_val=xe,train_param_mean=param_mean,train_param_std=param_std)
xe_initial = np.append(np.zeros(state_dim),xe[-2:])
ye_initial = Fwmodel(input_val=xe_initial)

plt.plot(ye,color='red')
plt.plot(ye_initial,color='blue')
plt.ylabel('ye')
plt.show()

#Add noise to ye
nn = 1  
ye_series = pd.Series(ye)
ye_n = np.zeros((nn,len(ye)))
shifts = np.random.choice(a=[-1,0,1],size=nn,replace=True) #uniform sampling
for i in range(nn):
    ye_n[i] = ye_series.shift(periods=shifts[i],fill_value=0.0).values
    ye_n[i] = ye_n[i]+np.random.normal(loc=0.0,scale=0.01*ye_n[i],size=len(ye_n[i]))
    
k = np.random.choice(range(nn))
plt.figure()
plt.plot(ye,color='red',label='true ye')
plt.plot(ye_n[k],color='blue',label='ye_n:{0}'.format(i))
plt.legend(loc='upper left')
plt.ylabel('ye')
plt.title("ye_n[{0}]".format(k))
plt.show()
#del(Fwmodel)
gc.collect()


################
#Set hyper-parameters for tf-agents
disc_factor = 1.0
sub_episode_length = 10 #number of time_steps in a sub-episode. 
episode_length = sub_episode_length*10  #an trajectory starts from the initial timestep and has no other initial timesteps
                                       #each trajectory will be split to multiple episodes
env_num = 10  #Number of parallel environments, each environment is used to generate an episode
alpha = 0.1 #regularization coefficient
param_alpha = 0.01 #regularization coefficient for actor_network
sub_episode_num = int(env_num*(episode_length/sub_episode_length)) #number of sub-episodes used for a single update of actor_net params
print("number of sub_episodes used for a single param update:", sub_episode_num)
#optimizer for training the actor_net in the REINFORCE_agent

#Learning Schedule = initial_lr * (C/(step+C))
class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, C):
        self.initial_learning_rate = initial_lr
        self.C = C
    def __call__(self, step):
        return self.initial_learning_rate*self.C/(self.C+step)
lr = lr_schedule(initial_lr=0.001, C=100000)   
opt = tf.keras.optimizers.legacy.SGD(learning_rate=lr)
train_step_num = 0


def compute_reward(state, xe=xe):
    x = np.append(state, xe[-2:], axis=0)
    R = np.sum(  (Fwmodel(x)-ye_n)**2  )/(800*nn)
    R = np.float32(1/(R+0.001))
    #R = np.float32(-R)
    return R

#Define the Environment for REINFORCE-OPT
class Env(py_environment.PyEnvironment):
    def __init__(self):
        '''The function to initialize an Env obj.
        '''
        #Specify the requirement for the value of action,
        #which is an argument of _step(self, action) that is later defined.
        #tf_agents.specs.BoundedArraySpec is a class.
        #_action_spec.check_array( arr ) returns true if arr conforms to the specification stored in _action_spec
        self._action_spec = array_spec.BoundedArraySpec(
                            shape=(state_dim,), dtype=np.float32, minimum=-0.1, maximum=0.1, name='action') #a_t is an 2darray
    
        #Specify the format requirement for observation,
        #i.e. the observable part of S_t, and it is stored in self._state
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(state_dim,), dtype=np.float32, name='observation') #default max and min is None
        self._state = np.array([0.0]*state_dim,dtype=np.float32)
        self._episode_ended = False
        self._step_counter = 0

    def action_spec(self):
        #return the format requirement for action
        return self._action_spec

    def observation_spec(self):
        #return the format requirement for observation
        return self._observation_spec

    def _reset(self):
        self._state = np.array([0.0]*state_dim,dtype=np.float32)  #initial state
        self._episode_ended = False
        self._step_counter = 0
        #return ts.restart(observation=np.array(self._state, dtype=np.float32))
        initial_r = np.float32(0.0)
        
        return ts.TimeStep(step_type=StepType.FIRST, 
                           reward=initial_r, 
                           discount=np.float32(disc_factor), 
                           observation=np.array(self._state, dtype=np.float32)
                           )
    
    def set_state(self,new_state):
        self._state = new_state
    
    
    def get_state(self):
        return self._state
    
    def _step(self, action):
        '''
        The function for the transtion from (S_t, A_t) to (R_{t+1}, S_{t+1}).
    
        Input.
        --------
        self: contain S_t.
        action: A_t.
    
        Output.
        --------
        an TimeStep obj, TimeStep(step_type_{t+1}, R_{t+1}, discount_{t+1}, observation S_{t+1})
        ''' 
        # Suppose that we are at the beginning of time t 
        
        ################## --- Determine whether we should end the episode.
        if self._episode_ended:  # its time-t value is set at the end of t-1
            return self.reset()
                
        ################# --- Compute S_{t+1} 
        self._state = self._state + action      
        
        ################# --- Compute R_{t+1}=R(S_t,A_t)
        R = compute_reward(self._state) 
        
        #print('observation:', self._state, 'action', action,  'reward', reward)
        self._step_counter +=1
        
        #Set conditions for termination
        if self._step_counter>=sub_episode_length-1:
            self._episode_ended = True  #value for t+1

        #Now we are at the end of time t, when self._episode_ended may have changed
        if self._episode_ended:
            #ts.termination(observation,reward,outer_dims=None): Returns a TimeStep obj with step_type set to StepType.LAST.
            return ts.termination(np.array(self._state, dtype=np.float32), reward=R)
        else:
            #ts.transition(observation,reward,discount,outer_dims=None): Returns 
            #a TimeStep obj with step_type set to StepType.MID.
            return ts.transition(np.array(self._state, dtype=np.float32), reward=R, discount=disc_factor)


#Create a sequence of environments and batch them, for later simulating them in external processes.
env_obj = Env()
parallel_env = ParallelPyEnvironment(env_constructors=[Env]*env_num, 
                                         start_serially=False,
                                         blocking=False,
                                         flatten=False
                                        )
#Use the wrapper to create two TFEnvironments obj. (so that parallel computation is enhanced)
train_env = tf_py_environment.TFPyEnvironment(parallel_env, check_dims=True) #instance
eval_env = tf_py_environment.TFPyEnvironment(env_obj, check_dims=False) #instance


#actor_distribution_network outputs a distribution
#it is a neural net which outputs the parameter (mean and sd, named as loc and scale) for a normal distribution
#actor_net = actor_distribution_network.ActorDistributionNetwork(   
actor_net = ActorDistributionNetwork(   
                                         train_env.observation_spec(),
                                         train_env.action_spec(),
                                         fc_layer_params=(128,64,16), #Hidden layers
                                         seed=0, #seed used for Keras kernal initializers for NormalProjectionNetwork.
                                         #discrete_projection_net=_categorical_projection_net
                                         continuous_projection_net=(NormalProjectionNetwork)
                                         )

#Create the  REINFORCE_agent
train_step_counter = tf.Variable(0)

REINFORCE_agent = reinforce_agent.ReinforceAgent(
        time_step_spec = train_env.time_step_spec(),
        action_spec = train_env.action_spec(),
        actor_network = actor_net,
        value_network = None,
        value_estimation_loss_coef = 0.2,
        optimizer = opt,
        advantage_fn = None,
        use_advantage_loss = False,
        gamma = 1.0, #discount factor for future returns
        normalize_returns = False, #The instruction says it's better to normalize
        gradient_clipping = None,
        entropy_regularization = None,
        train_step_counter = train_step_counter
        )
       
REINFORCE_agent.initialize()

eval_policy = REINFORCE_agent.policy
collect_policy = REINFORCE_agent.collect_policy

#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=10, max_step=100):
    '''
    This function perform 'policy' in 'environment' for a few episodes and 
    compute the average of sum-of-total-rewards.
    '''
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        step_count = 0

        while ((not time_step.is_last()) & (step_count<max_step)) :
            action_step = policy.action(time_step)  #output an action based on time_step.observation (S_t)
            time_step = environment.step(action_step.action)  #take the action and output the next state and reward
            episode_return += time_step.reward  #update the total rewards
            step_count += 1
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def extract_episode(traj_batch,epi_length,attr_name = 'observation'):
    """
    This function extract episodes (each episode consists of consecutive time_steps) from a batch of trajectories.
    Inputs.
    -----------
    traj_batch:replay_buffer.gather_all(), a batch of trajectories
    epi_length:int, number of time_steps in each extracted episode
    attr_name:str, specify which data from traj_batch to extract
    
    Outputs.
    -----------
    tf.constant(new_attr,dtype=attr.dtype), shape = [new_batch_size, epi_length, state_dim]
                                         or shape = [new_batch_size, epi_length]
    """
    attr = getattr(traj_batch,attr_name)
    original_batch_dim = attr.shape[0]
    traj_length = attr.shape[1]
    epi_num = int(traj_length/epi_length) #number of episodes out of each trajectory
    batch_dim = int(original_batch_dim*epi_num) #new batch_dim
    
    if len(attr.shape)==3:
        stat_dim = attr.shape[2]
        new_attr = np.zeros([batch_dim, epi_length, state_dim])
    else:
        new_attr = np.zeros([batch_dim, epi_length])
        
    for i in range(original_batch_dim):
        for j in range(epi_num):
            new_attr[i*epi_num+j] = attr[i,j*epi_length:(j+1)*epi_length].numpy()
        
    return tf.constant(new_attr,dtype=attr.dtype)


#################
#replay_buffer is used to store policy exploration data
#################
replay_buffer = rb.TFUniformReplayBuffer(
                data_spec = REINFORCE_agent.collect_data_spec,  # describe spec for a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                batch_size = env_num,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                         # One batch corresponds to one parallel environment
                max_length = episode_length*100    # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                    # if exceeding this number previous trajectories will be dropped
)

#test_buffer is used for evaluating a policy
test_buffer = rb.TFUniformReplayBuffer(
                data_spec= REINFORCE_agent.collect_data_spec,  # describe a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                batch_size= 1,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                                    # train_env.batch_size: The batch size expected for the actions and observations.  
                                                    # batch_size: Batch dimension of tensors when adding to buffer. 
                max_length = episode_length         # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                    # if exceeding this number previous trajectories will be dropped
)


#A driver uses an agent to perform its policy in the environment.
#The trajectory is saved in replay_buffer
collect_driver = DynamicEpisodeDriver(
                                     env = train_env, #train_env contains parallel environments (no.: env_num)
                                     policy = REINFORCE_agent.collect_policy,
                                     observers = [replay_buffer.add_batch],
                                     num_episodes = sub_episode_num   #SUM_i (number of episodes to be performed in the ith parallel environment)
                                    )

#For policy evaluation
test_driver = py_driver.PyDriver(
                                     env = eval_env, #PyEnvironment or TFEnvironment class
                                     policy = REINFORCE_agent.policy,
                                     observers = [test_buffer.add_batch],
                                     max_episodes=1, #optional. If provided, the data generation ends whenever
                                                      #either max_steps or max_episodes is reached.
                                     max_steps=sub_episode_length
                                )


######## Train REINFORCE_agent's actor_network multiple times.
update_num = 7500
eval_intv = 100 #number of updates required before each policy evaluation
eval_results = [] #for logging evaluation results
#num_eval_episodes = 10
max_reward = -1000.0
tolerence = 20 #If the eval_results does not improve within tolerence*eval_intv, then stop training.

tf.random.set_seed(0)
for n in range(0,update_num):
    #Generate Trajectories
    replay_buffer.clear()
    collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
    
    experience = replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
    rewards = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'reward') #shape=(sub_episode_num, sub_episode_length)
    observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)
    actions = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'action') #shape=(sub_episode_num, sub_episode_length, state_dim)
    step_types = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'step_type')
    discounts = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'discount')
    
    time_steps = ts.TimeStep(step_types,
                             tf.zeros_like(rewards),
                             tf.zeros_like(discounts),
                             observations
                            )
    
    rewards_sum = tf.reduce_sum(rewards, axis=1) #shape=(sub_episode_num,)
    
    with tf.GradientTape() as tape:
        #trainable parameters in the actor_network in REINFORCE_agent
        variables_to_train = REINFORCE_agent._actor_network.trainable_weights
    
        ###########Compute J_loss = -J
        actions_distribution = REINFORCE_agent.collect_policy.distribution(
                               time_steps, policy_state=None).action
    
        #log(pi(action|state)), shape = (batch_size, epsode_length)
        action_log_prob = common.log_probability(actions_distribution, 
                                                 actions,
                                                 REINFORCE_agent.action_spec)
    
        J = tf.reduce_sum(tf.reduce_sum(action_log_prob,axis=1)*rewards_sum)/sub_episode_num
        
        ###########Compute regularization loss from actor_net params
        regu_term = tf.reduce_sum(variables_to_train[0]**2)
        num = len(variables_to_train) #number of vectors in variables_to_train
        for i in range(1,num):
            regu_term += tf.reduce_sum(variables_to_train[i]**2)
        
        total = -J + param_alpha*regu_term
        
    #update parameters in the actor_network in the policy
    grads = tape.gradient(total, variables_to_train)
    grads_and_vars = list(zip(grads, variables_to_train))
    opt.apply_gradients(grads_and_vars=grads_and_vars)
    train_step_num += 1
    
    #Compute the mean performance
    obs_mean = np.mean(observations,axis=0) #shape: [traj-length, state_dim]
    rewards = []
    for i in range(sub_episode_length):
        rewards.append(compute_reward(obs_mean[i]))
        
    best_index = np.argmax(rewards)   #This is the best index for obs_mean, i.e., R(obs[best_index]) is largest
                                      #The best index for rewards is best_index-1
                                      
    
    
    #Evaluate policy performance and log result
    if train_step_num % eval_intv == 0:
        max_reward = rewards[best_index]
        best_obs = obs_mean[best_index]
        y_hat = Fwmodel(np.append(best_obs,xe[-2:]))
        best_rwd = compute_reward(best_obs)
        #Print and plot the best_obs       
        #print('best_obs',best_obs.round(3))
        print('best_index=',best_index)
        print("train_step no.=",train_step_num,
              "best step reward so far=",round(max_reward,4)
             )
        print('xe R2=', r2_score(xe[0:-2],best_obs).round(3),
              'ye R2=', r2_score(ye,y_hat).round(3))
        #print('episode of rewards', rewards.round(3))

        plt.figure()
        plt.plot(xe[0:-2],color='red',label='true xe')
        plt.plot(best_obs,color='blue',label='estimated xe')
        plt.legend(loc='upper left')
        plt.show()
            
        plt.figure()
        plt.plot(ye,color='red',label='true ye')
        plt.plot(y_hat,color='blue',label='estimated ye')
        plt.legend(loc='upper left')
        plt.ylabel('ye')
        plt.show()
        
        TrajectoryReward = np.sum(rewards)
        print('train_step_no. = {0}: Avg_traj_R = {1}'.format(train_step_num, TrajectoryReward))
        eval_results.append(TrajectoryReward)
        
        #Stopping Rule
        if len(eval_results)>=tolerence:
            if TrajectoryReward>0.45:
                break
        

#Plot training history
steps = range(0, train_step_num, eval_intv)
plt.figure(figsize=(12,6))
plt.plot(steps, eval_results)
plt.title('Performance of the Evolved Policy',size=15)
plt.ylabel('Sum of Rewards in a Generated Trajectory',size=15)
plt.xlabel("Number of Times " + r'$\theta$'+ " Gets Updated Update",size=15)
plt.tick_params(labelsize=15)
plt.savefig("chromatography-log.eps",bbox_inches='tight')


############Generate 10000 Trajectories 
traj_num = 1000
loop_num = int(traj_num/sub_episode_num)
residual_num = traj_num%sub_episode_num
estimates = np.zeros((traj_num,state_dim),dtype=np.float32)

for n in range(0,loop_num):
    #Generate 60 Trajectories
    replay_buffer.clear()
    collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
    experience = replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
    observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)=(60,10,64)
    estimates[n*sub_episode_num:(n+1)*sub_episode_num] = observations.numpy()[:,-1,:]

if residual_num>0:
    replay_buffer.clear()
    collect_driver.run()
    observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)=(60,10,64)
    estimates[-residual_num:] = observations[0:residual_num,-1,:]
    print(np.min(np.sum(np.abs(estimates),axis=1)))


#Bootstrapping Confidence Interval (CI) for Group1 Samples
B = 10000  #number of bootstrapping times 
bootstrap_mean_list = []
index = range(traj_num)
for i in range(B):
    bootstrap_index = np.random.choice(a=index,size=5,replace=True)
    bootstrap_sample = estimates[bootstrap_index]
    bootstrap_mean_list.append(np.mean(bootstrap_sample,axis=0))

gp1CI_lb = np.percentile(a=bootstrap_mean_list,q=0.5,axis=0) #CI lower bound for grp1
gp1CI_up = np.percentile(a=bootstrap_mean_list,q=99.5,axis=0) #CI upper bound for grp1
gp1CI_up = gp1CI_up*param_std[0:8]+param_mean[0:8]
gp1CI_lb = gp1CI_lb*param_std[0:8]+param_mean[0:8]
#gp1CI_lb = np.min(bootstrap_mean_list,axis=0) #CI lower bound for grp1
#gp1CI_up = np.max(bootstrap_mean_list,axis=0) #CI upper bound for grp1
print("Upper Bound of CI")
print(gp1CI_up) #param_std and param_mean are used to convert the estimates back to the original scale
print("Lower Bound of CI")
print(gp1CI_lb)


########plot
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'cm:italic:bold'

true_xe = xe[0:8]*param_std[0:8]+param_mean[0:8]
xe_est = np.mean(estimates,axis=0)
ye_est = Fwmodel(np.append(xe_est,xe[-2:]))
ye_R2 = r2_score(ye,ye_est).round(3)
print('ye R2=',ye_R2)
xe_est = xe_est*param_std[0:8]+param_mean[0:8]


plt.figure(figsize=(10,4))
plt.plot(ye,color='red',label=r"$f(\mathbf{x}_e)$")
plt.plot(ye_est,color='blue',marker='o',linestyle='--', markersize=4.5, label=r"$f(\bar{\mathbf{x}}_{T-1})$")
plt.legend(loc='upper left',fontsize=15)
plt.tick_params(labelsize=15)
#ax[1].set_xlim(-0.5,85)
plt.savefig("chromatography.eps",transparent=True,bbox_inches='tight')
plt.show()

