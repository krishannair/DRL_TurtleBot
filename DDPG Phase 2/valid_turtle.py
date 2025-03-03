import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from env import Env
from helper import plot
import rospy
import argparse
import time

class ReplayBuffer(object):
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, new_state, done):
        #If mem_cntr exceeds max storage storage space then overwrite the oldest data
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        #max_mem will be equal to mem_cntr when the storage space has never been completely 
        #filled else it will be equal to mem_size tell us the useable data before completely filling
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals
    
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        #Learning rate = beta
        #n_actions here is not the number of actions but the dimensionality of the action like here 2D therefore 2
        #chkpt_dir is the location where the models will be saved
        super(CriticNetwork, self).__init__()
        #Initializing the nn.Module
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.pth')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # *self.input_dims will unpack the list so that we can have variable input size to this layer
        f1 = 1/ np.sqrt(self.fc1.weight.data.size()[0])
        # We initialize with the 1/square root of the number of inputs as then the gradient wont 
        # become very large or very small 
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        # Initializing the weights with the value of -f1 to f1 uniform distribution
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        # Initializing the biases
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        # Initializing the Normalization in the Feature dimension

        # Doing the same for the next Linear layer in the neural network
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # fc2_dims will have to be equal to n_actions else there will be a mismatch in layer output and input
        
        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003  # constant initialization for making gradient not start with really high or low values
        self.q = nn.Linear(self.fc2_dims, 1)    # 1 output parameter
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        # Initializing the optimizer 
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Selecting the cpu to perform the actions on the tensors and the objects
        self.device = T.device('cpu')
        
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)   # fully connected layer
        state_value = self.bn1(state_value) # batch normalization
        state_value = F.relu(state_value)   # Activation Layer rectified linear unit function
        state_value = self.fc2(state_value) # fully connected layer
        state_value = self.bn2(state_value) # batch normalization
        
        #action_value is being calculated like a different neural net not the same as the state_value
        action_value = F.relu(self.action_value(action)) #Ensure non negativity in the output action value
        # Getting action from the actor network to calculate action_value
        state_action_value = F.relu(T.add(state_value, action_value)) #Ensure non negativity in the output Q-value
        state_action_value = self.q(state_action_value)
        
        return state_action_value
    
    # Function to save model
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    
    # Function to load saved model
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.pth')
        
        # Same as done with the critic network initializing layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims, self.fc2_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        # Layers to Decide what action to do based on the state given
        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))
        # Tanh used to bound action from -1 to 1 ReLu doesnt bound...
        return x
        
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        # We can call an object as a function 
        # noise = OUActionNoise()
        # noise()
        # Noise reverts to mean at the rate we choose and the mean also we choose(mu)
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        # Saving the value of Noise for the next time step so that its continuous
        self.x_prev = x
        return x
    
    def reset(self):
        # The shape of the noise is equal to shape of mu
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=2000000, layer1_size=400,
                 layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.max_action
        self.min_action = env.min_action
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='Actor')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='TargetActor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions,
                                    name='Critic')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions,
                                           name='TargetCritic')
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self,observation):
        self.actor.eval()

            # Evaluation mode will make sure that the actions provided by actor are consistent
            # till the actor reports the actions to be performed for the state and not get trained
            # and change the actions inconsistently
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        # Adding the noise for exploration
        mu_prime = mu # Removed Noise from code for validation
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        # Putting agent back to training mode
        self.actor.train()
        # Returning the action chosen as a numpy array
        return mu_prime.cpu().detach().numpy()  

    def update_network_parameters(self, tau=None):
            #Tau is what determines if it is a soft update or hard update
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        # Soft Update of each parameter of Actor and Critic Networks
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                (1 - tau) * target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
        
def ddpg(n_games):
    rospy.init_node('ddpg')
    print("Validation Started")
    env = Env(False)
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[364], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)
    np.random.seed(0)
    past_action = np.array([0., 0.])
    agent.load_models()
    score_history = []
    # print("Before going through eps")
    for i in range(n_games):
        done = False
        score = 0
        # print("before env reset")
        obs = env.reset()
        last_goal_time = time.time()
        # print("inside episode")
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, goal, _ = env.step(act, past_action)
            # Done if succeeds or if time runs out
                # we learn on every step (temporal difference learning method)
                # instead of at the end of every episode (Monte Carlo method)
            if(goal == True):
                last_goal_time = time.time()
            curr_time = time.time()
            # if(curr_time - last_goal_time > 40):
            #     reward -= 250
            #     done = True
            score += reward
            obs = new_state
            past_action = act
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score)
        filename = 'Turtle_valid_' + str(n_games) + '_games.png'
    print("Validation Finished!")
    plotLearning(score_history, filename, window=100)

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.figure()
    plt.ylabel('Score')       
    plt.xlabel('Episode')                     
    plt.plot(x, running_avg)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()  # Add this line to display the plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-eps', type=int, default=10, help='Number of validation Episodes (default: 10)')  
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print("Validation Starting")
    ddpg(n_games=args.eps)
