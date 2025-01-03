import numpy as np
import os, json
from tqdm import tqdm
from collections import deque
# self-defined packages
from codebase import utility
from codebase.probmats import ProbMat
from codebase.bcienv import SimulatedEnv
############
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
mu1 = [0.9, 1.2, 1.5][job_id%3]
mu = [0, mu1]; sigma = [1, 1]
probmat = ProbMat(mu, sigma)
result = {'accu': [], 'util': [], 'time': []}
############
for n_seq in tqdm(range(1, 16)):
    env = SimulatedEnv(probmat, t_flash=0.8/60, t_chr=5/60, score_mu=mu, score_sigma=sigma, p0=0)
    obs, info = env.reset()
    #
    rewards = []
    #
    for _ in range(50):
        actions = np.concatenate([np.random.permutation(np.arange(12)) for _ in range(n_seq)])
        actions = list(actions)+[12]
        for action in actions:
            # update gp
            obs_old = obs
            obs, reward, _, _, info = env.step(action)
            #
            if action == 12:
                rewards.append(reward)
                break
    # summary results
    n_correct, n_wrong = np.sum(np.array(rewards)>0), np.sum(np.array(rewards)<0)
    result['accu'].append(n_correct/len(rewards))
    result['util'].append(utility.cal_utility(n_correct, n_wrong, env.time))
    result['time'].append(env.time/env.current_chr_id)
#
with open(f'results/sim1/fs_{mu1}_{job_id//3}', "w") as file:
    json.dump(result, file, indent=4)
############
############
env = SimulatedEnv(probmat, t_flash=0.8/60, t_chr=5/60, score_mu=mu, score_sigma=sigma, p0=0)
obs, info = env.reset()
#
result = {'accu': [], 'util': [], 'time': []}
rewards = []
#
for _ in tqdm(range(50)):
    obs_hist = deque(maxlen=2)
    for seq in range(15):
        for action in np.random.permutation(np.arange(12)):
            # update gp
            obs, reward, _, _, info = env.step(action)
        obs_hist.append(obs)
        if len(obs_hist)==2:
            cond1 = obs_hist[0]['certainty_scores'].max() > 0.5 and obs_hist[1]['certainty_scores'].max() > obs_hist[0]['certainty_scores'].max()
            cond2 = obs_hist[0]['certainty_scores'].argmax() == obs_hist[1]['certainty_scores'].argmax()
            if cond1 and cond2:
                obs, reward, _, _, info = env.step(12)
                rewards.append(reward)
                break
# summary results
n_correct, n_wrong = np.sum(np.array(rewards)>0), np.sum(np.array(rewards)<0)
result['accu'].append(n_correct/len(rewards))
result['util'].append(utility.cal_utility(n_correct, n_wrong, env.time))
result['time'].append(env.time/env.current_chr_id)
#
with open(f'results/sim1/agree_{mu1}_{job_id//3}', "w") as file:
    json.dump(result, file, indent=4)
