import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import multiprocessing
import torch
import warnings

class Runner:
    def __init__(self, env, args):
        self.env = env
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.writer = SummaryWriter(log_dir='runs/{}/exp2'.format(args.alg))
        self.indicators = []
        self.win_rates = []
        self.episodic_aois = []
        self.episode_rewards = []
        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                indicator, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                # self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                # self.plt(num)
                evaluate_steps += 1
                self.writer.add_scalar('episodic_aoi', indicator['episodic_aoi'],num * self.args.n_steps + time_steps)
                self.writer.add_scalar('T-AoI', indicator['T-AoI'],num * self.args.n_steps + time_steps)
                self.writer.add_scalar('collection_ratio', indicator['collection_ratio'],num * self.args.n_steps + time_steps)
                self.writer.add_scalar('episode_reward', episode_reward, num * self.args.n_steps + time_steps)
        
            episodes = []
            # results = []
            # manager = multiprocessing.Manager()
            # pool = multiprocessing.Pool(processes=self.args.n_workers)
            # for i in range(self.args.n_workers):
            #     result = pool.apply_async(self.rolloutWorker.generate_episode, args=(i,))
            #     results.append(result)
            # print('--------begin waiting-------')
            # pool.close()
            # pool.join()
            # print('--------end waiting-------')
            # # for i in range(queue.qsize()):
            # #     episode, _, _, steps = queue.get()
            # #     time_steps += steps
            # #     episodes.append(episode)
            
            # with torch.no_grad():
            #     for result in results:
            #         episode, _, _, steps = result.get()
            #         episodes.append(episode)
            #         time_steps += steps

            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps

            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        indicator, episode_reward = self.evaluate()
        # print('win_rate is ', win_rate)
        # self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.indicators.append(indicator)
        self.episodic_aois.append(indicator['episodic_aoi'])
        # self.plt(num)

    def evaluate(self):
        # win_number = 0
        episode_rewards = 0
        retdict = {'collection_ratio':[], 'violation_ratio':[],	'episodic_aoi':[], 'T-AoI':[], 'consumption_ratio':[]}
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, r, _ = self.rolloutWorker.generate_episode(0, evaluate=True)
            # episode_reward, r, _ = self.rolloutWorker.generate_episode(0, evaluate=False)
            episode_rewards += episode_reward
            retdict['collection_ratio'].append(r['collection_ratio'])
            retdict['violation_ratio'].append(r['violation_ratio'])
            retdict['episodic_aoi'].append(r['episodic_aoi'])
            retdict['T-AoI'].append(r['T-AoI'])
            retdict['consumption_ratio'].append(r['consumption_ratio'])
            # if win_tag:
            #     win_number += 1
            # get average rate
        retdict['collection_ratio'] = np.mean(retdict['collection_ratio'])
        retdict['violation_ratio'] = np.mean(retdict['violation_ratio'])
        retdict['episodic_aoi'] = np.mean(retdict['episodic_aoi'])
        retdict['T-AoI'] = np.mean(retdict['T-AoI'])
        retdict['consumption_ratio'] = np.mean(retdict['consumption_ratio'])

        print('-------episodic_aoi----------', retdict['episodic_aoi'])
        print('-------episode_rewards----------', episode_rewards / self.args.evaluate_epoch)
        return retdict, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.episodic_aois)), self.episodic_aois)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()









