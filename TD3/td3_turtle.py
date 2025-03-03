import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from matplotlib import pyplot as plt
from env import Env
from helper import plot
import rospy
import argparse
import time

class ReplayBuffer():
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
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/td3'):
        #Learning rate = beta
        #n_actions here is not the number of actions but the dimensionality of the action like here 2D therefore 2
        #chkpt_dir is the location where the models will be saved
        super(CriticNetwork, self).__init__()
        #Initializing the nn.Module
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3.pth')

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
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        
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
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3.pth')
        
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
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        
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

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2,
            n_actions=2, max_size=2000000, layer1_size=400,
            layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.max_action
        self.min_action = env.min_action
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0

        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                        layer2_size, n_actions=n_actions, name='critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                    layer2_size, n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                layer2_size, n_actions=n_actions, name='target_critic_2')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()

            # Evaluation mode will make sure that the actions provided by actor are consistent
            # till the actor reports the actions to be performed for the state and not get trained
            # and change the actions inconsistently
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        # Adding the noise for exploration
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        # Putting agent back to training mode
        self.actor.train()
        # Returning the action chosen as a numpy array
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # Not ready to learn if observations till the batch size is not made
        # Else sampling from the buffer
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        # Converting them to tensor using the CPU as we gave device as that before
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)

        self.target_actor.eval()    # evaluation mode (testing model)
        self.target_critic_1.eval()
        self.critic_1.eval()
        self.target_critic_2.eval()
        self.critic_2.eval()

        target_actions = self.target_actor.forward(new_state)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0], 
                                self.max_action[0])
        
        target_critic_1_value = self.target_critic_1.forward(new_state, target_actions)
        target_critic_2_value = self.target_critic_2.forward(new_state, target_actions)

        critic_1_value = self.critic_1.forward(state, action)
        critic_2_value = self.critic_2.forward(state, action)

        # target_critic_1_value[done] = 0.0
        # target_critic_2_value[done] = 0.0

        # target_critic_1_value = target_critic_1_value.view(-1)
        # target_critic_2_value = target_critic_2_value.view(-1)

        critic_value_ = T.min(target_critic_1_value, target_critic_2_value)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic_1.device)
        target = target.view(self.batch_size, 1)
        # target = reward + self.gamma*critic_value_
        # target = target.view(self.batch_size, 1)

        self.critic_1.train()
        self.critic_2.train()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()


        target_critic_1_value_loss = F.mse_loss(target, critic_1_value)
        target_critic_2_value_loss = F.mse_loss(target, critic_2_value)
        critic_loss = target_critic_1_value_loss + target_critic_2_value_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()


        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
        self.critic_1.eval() 
        self.critic_2.eval() 

        self.actor.optimizer.zero_grad()
        self.actor.train()
        actor_q1_loss = -self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

def td3(n_games):
    rospy.init_node('td3')
    start_time = time.time()
    print("Training Started")
    env = Env(False)
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[364], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)
    best_score = -10000
    np.random.seed(0)
    past_action = np.array([0., 0.])
    #agent.load_models()
    score_history = []
    eps_training_times = []
    dist_travelled = []
    # print("Before going through eps")
    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()
        eps_start_time = time.time()
        last_goal_time = time.time()
        dist_one_eps = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, goal, dist= env.step(act, past_action)
            dist_one_eps = dist_one_eps + dist
            # Done if succeeds or if time runs out
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
                # we learn on every step (temporal difference learning method)
                # instead of at the end of every episode (Monte Carlo method)
            if(goal == True):
                last_goal_time = time.time()
            curr_time = time.time()
            if(curr_time - last_goal_time > 25):
                score -= 1000
                done = True
            score += reward
            obs = new_state
            past_action = act
        dist_travelled.append(dist_one_eps)
        eps_end_time = time.time()
        eps_training_times.append(eps_end_time - eps_start_time)
        score_history.append(score)
        with open("scores.txt", 'a') as file2:
            file2.write(str(score) + " ")
        with open("dist.txt", 'a') as file3:
            file3.write(str(dist_one_eps) + " ")
        with open("time.txt", 'a') as file4:
            file4.write(str(eps_end_time-eps_start_time) + " ")
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
        print('episode ', i, 'score %.2f' % score,
            '100 game average %.2f' % avg_score)
        if i % 25 == 0:
            agent.save_models()
        filename = 'Turtle_td3_' + str(n_games) + '_games.png'
    print("Training Finished!")
    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: ", training_time, " seconds")
    plotter(dist_travelled, filename='Episode_td3_train_dist.png', ylab = 'Units', window = 100)
    plotter(eps_training_times, filename='Episode_td3_train_time_curve.png', ylab = 'Seconds', window = 100)
    plotLearning(score_history, filename, window=100)
    print("Best Score = ", best_score)
    with open("training_time.txt", 'w') as file:
        file.write("Training Time = " + str(training_time) + "\n " + "Best Score = " + str(best_score))
    print("Float data saved to training_time.txt")
    agent.save_models()
    T.save(agent.actor.state_dict(), 'tmp/td3/final_td3_actor.pth')
    T.save(agent.critic_1.state_dict(), 'tmp/td3/final_td3_critic_1.pth')
    T.save(agent.critic_2.state_dict(), 'tmp/td3/final_td3_critic_2.pth')
    T.save(agent.target_actor.state_dict(), 'tmp/td3/final_td3_actor_target.pth')
    T.save(agent.target_critic_1.state_dict(), 'tmp/td3/final_td3_critic_target_1.pth')
    T.save(agent.critic_1.state_dict(), 'tmp/td3/final_td3_critic_target_2.pth')
    print("Saved Final Models")


def plotter(scores, filename, x=None, ylab='Units', window = 5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.figure()
    plt.ylabel(ylab)       
    plt.xlabel('Episode')                     
    plt.plot(x, running_avg)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()  # Add this line to display the plot

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
    parser.add_argument('-eps', type=int, default=1000, help='Number of training Episodes (default: 1000)')  
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print("Training Starting")
    td3(n_games=args.eps)
