import numpy as np
from tqdm import tqdm
import json, os
from collections import deque
# self-defined packages
from codebase import utility
from codebase.probmats import ProbMat
from codebase.bcienv import SimulatedEnv
from codebase.agent import Agent
############
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
mu1 = [0.9, 1.2, 1.5][job_id%3]
mu = [0, mu1]; sigma = [1, 1]
probmat = ProbMat(mu, sigma)
result = {'accu': [], 'util': [], 'time': []}
n_delay = 3
############
agent = Agent()
#
env = SimulatedEnv(probmat, t_flash=0.2/60, t_chr=5/60, score_mu=mu, score_sigma=sigma, p0=0.5)
obs, info = env.reset()
for _ in tqdm(range(5000)):
    actions = np.concatenate([np.random.permutation(np.arange(12)) for _ in range(15)])
    actions = list(actions)+[12]
    obs_hist = deque(maxlen=1+n_delay)
    for action_transformed in actions:
        obs_hist.append(obs)
        #
        action = agent.sample_action(obs_hist[0])
        if action == 1:
            action_transformed = 12
        #
        obs, reward, _, _, info = env.step(action_transformed)
        agent.add_data(obs_hist[-1], action_transformed, reward, obs)
        #
        if action_transformed == 12:
            agent.update_actor()
            agent.update_critic()
            break
############
env = SimulatedEnv(probmat, t_flash=0.2/60, t_chr=5/60, score_mu=mu, score_sigma=sigma, p0=0.5)
obs, info = env.reset()
rewards = []
for _ in tqdm(range(50)):
    actions = np.concatenate([np.random.permutation(np.arange(12)) for _ in range(15)])
    actions = list(actions)+[12]
    obs_hist = deque(maxlen=1+n_delay)
    for action_transformed in actions:
        obs_hist.append(obs)
        #
        action = agent.sample_action(obs_hist[0])
        if action == 1:
            action_transformed = 12
        #
        obs, reward, _, _, info = env.step(action_transformed)
        #
        if action_transformed == 12:
            rewards.append(reward)
            break
# summary results
n_correct, n_wrong = np.sum(np.array(rewards)>0), np.sum(np.array(rewards)<0)
result['accu'].append(n_correct/len(rewards))
result['util'].append(utility.cal_utility(n_correct, n_wrong, env.time))
result['time'].append(env.time/env.current_chr_id)
############
agent.fit_transition_model()
env = SimulatedEnv(probmat, t_flash=0.2/60, t_chr=5/60, score_mu=mu, score_sigma=sigma, p0=0.5)
obs, info = env.reset()
#
for _ in tqdm(range(2000)):
    obs_hist = deque(maxlen=1+n_delay)
    action_transformed_hist = deque(maxlen=n_delay)
    while True:
        obs_hist.append(obs)
        #
        action = agent.sample_action(obs_hist[0])
        cond = obs['current_flash_id'] < 12*15 and action==0
        action_transformed = agent.random_shoot_action(
            obs_hist[0]['log_certainty_scores'], list(action_transformed_hist), 10) if cond else 12
        # take the action
        obs, reward, _, _, info = env.step(action_transformed)
        action_transformed_hist.append(action_transformed)
        agent.add_data(obs_hist[-1], action_transformed, reward, obs)
        #
        if obs['current_flash_id'] == 0:
            agent.update_actor()
            agent.update_critic()
            break
############
env = SimulatedEnv(probmat, t_flash=0.2/60, t_chr=5/60, score_mu=mu, score_sigma=sigma, p0=0.5)
obs, info = env.reset()
rewards = []
#
for _ in tqdm(range(50)):
    obs_hist = deque(maxlen=1+n_delay)
    action_transformed_hist = deque(maxlen=n_delay)
    while True:
        obs_hist.append(obs)
        #
        action = agent.sample_action(obs_hist[0])
        cond = obs['current_flash_id'] < 12*15 and action==0
        action_transformed = agent.random_shoot_action(
            obs_hist[0]['log_certainty_scores'], list(action_transformed_hist), 10) if cond else 12
        # take the action
        obs, reward, _, _, info = env.step(action_transformed)
        action_transformed_hist.append(action_transformed)
        #
        if obs['current_flash_id'] == 0:
            rewards.append(reward)
            break
# summary results
n_correct, n_wrong = np.sum(np.array(rewards)>0), np.sum(np.array(rewards)<0)
result['accu'].append(n_correct/len(rewards))
result['util'].append(utility.cal_utility(n_correct, n_wrong, env.time))
result['time'].append(env.time/env.current_chr_id)
#
with open(f'results/sim2/acrs_{mu1}_{job_id//3}', "w") as file:
    json.dump(result, file, indent=4)
