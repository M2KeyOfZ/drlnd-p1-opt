########################################################################
#                                                                      #
# This code is adapted from yingweiy's excellent GitHub repo:          #
#   https://github.com/yingweiy/drlnd_project1_navigation.git          #
#                                                                      #
########################################################################

import time
import torch
import numpy as np
from collections import deque
from dqn_agent import Agent


class DQN():
    def __init__(self, state_size, action_size, env):
        self.agent = Agent(state_size=state_size, action_size=action_size, seed=0)
        self.env = env
        self.saved_network = 'VisualBanana_DQN_chkpt.pth'

    def train(self, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
              score_window_size=100, target_score=13.0, save=True, verbose=True):
        """Deep Q-Learning.

            Params
            ======
                n_episodes  (int): max. number of training episodes
                max_t       (int): max. number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end   (float): min. value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
            """
        
        moving_avgs   = []         # list containing moving average scores (over last 100 episodes)
        scores        = []         # list containing scores from each episode
        scores_window = deque(maxlen=score_window_size)  # last score_window_size scores
        eps           = eps_start  # initialize epsilon
        save12 = False

        start = time.time()
        for i_episode in range(1, n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state  = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)          # save most recent score
            scores.append(score)                 # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            avg_score = np.mean(scores_window)
            moving_avgs.append(avg_score)

            if (avg_score >= 13.0) and not save12:
                torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)
                np.save('VisualBanana_Scores.npy', np.array(scores))
                save12 = True

            if (avg_score >= target_score) and (i_episode > 100):
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'\
                        .format(i_episode-100, np.mean(scores_window)))
                self.solved = True
                if save:
                    torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)
                break

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if (i_episode % 100 == 0):
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            if (i_episode % 100 == 0):
                end = time.time()
                elapsed = (end-start) / 60.0
                print('\tElapsed: {:3.2f} mins.'.format(elapsed))

        if save:
            torch.save(self.agent.qnetwork_local.state_dict(), self.saved_network)

        end = time.time()
        elapsed = (end-start) / 60.0
        print('\n*** TOTAL ELAPSED: {:3.2f} mins. ***'.format(elapsed))
            
        return scores, moving_avgs
