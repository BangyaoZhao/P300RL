import numpy as np
import torch, random
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .gp.BayesGP import GP
from . import utility

class PolicyNet(nn.Module):
    """Parametrized Policy Network."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 256), 
            nn.ReLU(),     
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = torch.log(x)
        return self.layers(x)

class CriticNet(nn.Module):
    """Parametrized Policy Network."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 256), 
            nn.ReLU(),
            nn.Linear(256, 1)   
        )

    def forward(self, x):
        x = torch.log(x)
        return self.layers(x)

class Agent:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.critic_net = CriticNet()
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=0.01)
        self.replay_buffer = deque(maxlen=1000)
        self.discount_rate = 0.5
        self.mistake_penalty = np.log(36-1)/np.log(2)

    def sample_action(self, obs):
        x = torch.tensor([[obs['certainty_scores'].max()]], dtype=torch.float32)
        with torch.no_grad():
            logit = self.policy_net(x)[0]
        rv = torch.distributions.Bernoulli(logits=logit)
        action = int(rv.sample())
        # 0 not stop, 1 stop
        return action
    
    def add_data(self, state, action, reward, next_state):
        if reward < -1:
            reward -= self.mistake_penalty
        self.replay_buffer.append((state, action, reward, next_state))
    
    def sample_data(self, batch_ratio=0.2):
        batch_size = round(1+self.replay_buffer.maxlen*batch_ratio)
        batch = random.choices(self.replay_buffer, k=batch_size)
        states, actions, rewards, next_states = zip(*batch)
        actions = [action==12 for action in actions]
        #
        maxscores = torch.tensor([x['certainty_scores'].max() for x in states]).reshape([-1, 1]).float()
        next_maxscores = torch.tensor([x['certainty_scores'].max() for x in next_states]).reshape([-1, 1]).float()
        times = torch.tensor([x['time'] for x in states]).reshape([-1, 1]).float()
        next_times = torch.tensor([x['time'] for x in next_states]).reshape([-1, 1]).float()
        rewards = torch.tensor(rewards).reshape([-1, 1]).float()
        actions = torch.tensor(actions, dtype=torch.int64).reshape([-1, 1])
        return (maxscores, times), actions, rewards, (next_maxscores, next_times)
    
    def update_actor(self, batch_ratio=0.2):
        (maxscores, times), actions, rewards, (next_maxscores, next_times) = self.sample_data(batch_ratio)
        with torch.no_grad():
            targets = rewards + np.exp(-self.discount_rate*(next_times-times))*self.critic_net(next_maxscores)
            values = self.critic_net(maxscores)
            advantages = targets.detach() - values.detach()
        logits = (2*actions-1)*self.policy_net(maxscores)
        logprobs = F.logsigmoid(logits)
        # update
        actor_loss = -torch.mean(logprobs*advantages)
        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()
    
    def update_critic(self, batch_ratio=0.2):
        (maxscores, times), actions, rewards, (next_maxscores, next_times) = self.sample_data(batch_ratio)
        with torch.no_grad():
            targets = rewards + np.exp(-self.discount_rate*(next_times-times))*self.critic_net(next_maxscores)
        values = self.critic_net(maxscores)
        # update
        advantages = targets - values
        critic_loss = torch.mean(torch.square(advantages))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
    
    def visualize_critic(self, ax):
        x = torch.tensor(np.linspace(1/36, 1, 100).reshape([-1, 1]), dtype=torch.float32)
        with torch.no_grad():
            y = self.critic_net(x).numpy()
        ax.plot(x, y)
    
    def visualize_policy(self, ax):
        x = torch.tensor(np.linspace(1/36, 1, 100).reshape([-1, 1]), dtype=torch.float32)
        with torch.no_grad():
            y = F.sigmoid(self.policy_net(x)).numpy()
        ax.plot(x, self.discount_rate*y)
    
    def fit_transition_model(self):
        data = []
        for trans in self.replay_buffer:
            logpmats = [trans[i]['log_certainty_scores'] for i in [0, 3]]
            action = trans[1]
            if action != 12:
                if action <= 5:
                    logps = [logpmat[action, :] for logpmat in logpmats]
                else:
                    logps = [logpmat[:, action-6] for logpmat in logpmats]
                logits = [utility.logit_exp(utility.log_sum_exp(logps[i])) for i in range(2)]
                data.append([np.sum(np.exp(logps[0])), logits[1] - logits[0]])
        x, y = zip(*data)
        x, y = np.array(x), np.array(y)
        #
        self.transition_model = GP(x, y)
        self.transition_model.update(1000)
    
    def random_shoot(self, logpmat, actions, n):
        if n != None:
            return np.stack([self.random_shoot(logpmat, actions, None) for _ in range(n)])
        logpmat = self.transition_model.simulate(logpmat, actions)
        pmaxs = []
        for last_action in range(12):
            logpmat1 = self.transition_model.simulate(logpmat, [last_action])
            pmaxs.append(np.exp(np.max(logpmat1)))
        pmaxs = torch.tensor(np.reshape(pmaxs, [-1, 1]), dtype=torch.float32)
        #
        with torch.no_grad():
            values = self.critic_net(pmaxs).numpy().flatten()
        #
        if len(actions) != 0 and logpmat.max() > 0.5:
            i, j = logpmat.argmax()//6, logpmat.argmax()%6+6
            if actions[-1] in [i, j]:
                values[i] = -10000
                values[j] = -10000
        return values
        
    
    def random_shoot_action(self, logpmat, actions, n = 10):
        sim_values = self.random_shoot(logpmat, actions, n)
        sim_values = np.mean(sim_values, axis=0)
        return sim_values.argmax()
        
         
         
        
    
