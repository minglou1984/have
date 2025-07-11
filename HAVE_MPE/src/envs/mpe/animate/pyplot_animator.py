import numpy as np

# for plot
import logging

logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import contextlib

"""
TODO
Adapt to mpe (currently is the uat animator)
"""


class MPEAnimator(animation.TimedAnimation):

    def __init__(self,
                 agent_positions,
                 landmark_positions,
                 episode_rewards,
                 mask_agents=False):

        # general parameters
        self.frames = (agent_positions.shape[1])
        self.n_agents = len(agent_positions)
        self.n_landmarks = len(landmark_positions)
        self.lags = self.frames
        self.fontsize = 18

        self.agent_positions = agent_positions
        self.landmark_positions = landmark_positions
        self.episode_rewards = episode_rewards

        # create the subplots
        self.fig = plt.figure(figsize=(13, 5))
        plt.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.17, wspace=0.22)  # Border distance
        self.ax_episode = self.fig.add_subplot(1, 2, 1)
        # plt.xticks(fontsize=self.fontsize)
        # plt.yticks(fontsize=self.fontsize)

        self.ax_reward = self.fig.add_subplot(1, 2, 2)
        # plt.xticks(fontsize=self.fontsize)
        # plt.yticks(fontsize=self.fontsize)

        # colors
        # color_cycle = sns.color_palette('tab10', 10)
        color_cycle = ['#D62728', '#2A9F2A', '#FF7F0E', '#1B75B4', '#9467BD', '#7F7F7F', '#E377C2', '#BBBC1F'
            , '#2A9F2A', '#FF7F0E', '#1B75B4', '#9467BD', '#7F7F7F', '#E377C2', '#BBBC1F'
            , '#2A9F2A', '#FF7F0E', '#1B75B4', '#9467BD', '#7F7F7F', '#E377C2', '#BBBC1F'
            , '#2A9F2A', '#FF7F0E', '#1B75B4', '#9467BD', '#7F7F7F', '#E377C2', '#BBBC1F'
            , '#2A9F2A', '#FF7F0E', '#1B75B4', '#9467BD', '#7F7F7F', '#E377C2',
                       '#BBBC1F']  # red, green, yellow, blue, purple
        self.agent_colors = color_cycle  # cm.Dark2.colors # green-blue, yellow, purple, pink, shallow green
        self.landmark_colors = color_cycle  # cm.Dark2.colors # [cm.summer(l*10) for l in range(self.n_landmarks)] # pastl greens

        # init the lines
        self.lines_episode = self._init_episode_animation(self.ax_episode)
        self.lines_reward = self._init_reward_animation(self.ax_reward)

        animation.TimedAnimation.__init__(self, self.fig, interval=100, blit=True)

    def save_animation(self, savepath='episode'):
        with contextlib.redirect_stdout(None):
            self.save(savepath + '.gif')
            self.fig.savefig(savepath + '.png')
            # self.fig.savefig(savepath + '.svg')

    def _episode_update(self, data, line, frame, lags, name=None):
        line.set_data(data[max(0, frame - lags):frame, 0], data[max(0, frame - lags):frame, 1])
        if name is not None:
            line.set_label(name)

    def _frameline_update(self, data, line, frame, name=None):
        line.set_data(np.arange(0, frame + 1), data[:frame + 1])

        if name is not None:
            line.set_label(name)

    def _draw_frame(self, frame):

        # Update the episode subplot
        line_episode = 0
        # update agents heads
        for n in range(self.n_agents):
            self._episode_update(self.agent_positions[n], self.lines_episode[line_episode], frame, 1, f'Agent {n + 1}')
            line_episode += 1

        # update agents trajectories
        for n in range(self.n_agents):
            self._episode_update(self.agent_positions[n], self.lines_episode[line_episode], max(0, frame - 1),
                                 self.lags)
            line_episode += 1

        # landmark real positions
        for n in range(self.n_landmarks):
            self._episode_update(self.landmark_positions[n], self.lines_episode[line_episode], frame, self.lags,
                                 f'Landmark {n + 1}')
            line_episode += 1

        # Update the reward subplot
        self._frameline_update(self.episode_rewards, self.lines_reward[0], frame)

        self._drawn_artists = self.lines_episode + self.lines_reward

    def _init_episode_animation(self, ax):
        # retrieve the episode dimensions
        x_max = max(self.agent_positions[:, :, 0].max(),
                    self.landmark_positions[:, :, 0].max())

        x_min = min(self.agent_positions[:, :, 0].min(),
                    self.landmark_positions[:, :, 0].min())

        y_max = max(self.agent_positions[:, :, 1].max(),
                    self.landmark_positions[:, :, 1].max())

        y_min = min(self.agent_positions[:, :, 1].min(),
                    self.landmark_positions[:, :, 1].min())

        abs_min = min(x_min, y_min)
        abs_max = max(x_max, y_max)

        ax.set_xlim(-1 - 0.1, 1 + 0.1)
        ax.tick_params(axis='x', labelsize=self.fontsize, width=2, length=8)
        ax.set_xlabel('X Position', fontsize=self.fontsize, labelpad=10)
        ax.set_ylim(-1 - 0.1, 1 + 0.1)
        ax.tick_params(axis='y', labelsize=self.fontsize, width=2, length=8)
        ax.set_ylabel('Y Position', fontsize=self.fontsize, labelpad=0)

        # ax.set_title('Episode', fontsize=self.fontsize, pad=10)

        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        x_major_locator = MultipleLocator(0.5)
        y_major_locator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        # ax.legend(fontsize=self.fontsize-2)
        # remove frame
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        # lines:
        # 1. agent head
        # 2. agent trajectory
        # 3. landmark real
        lines = [ax.plot([], [], 'o', color=self.agent_colors[a % 8], alpha=0.8, markersize=14)[0] for a in
                 range(self.n_agents)] + \
                [ax.plot([], [], 'o', color=self.agent_colors[a % 8], alpha=0.4, markersize=12)[0] for a in
                 range(self.n_agents)] + \
                [ax.plot([], [], 's', color=self.landmark_colors[l % 8], alpha=0.8, markersize=14)[0] for l in
                 range(self.n_landmarks)]
        return lines

    def _init_reward_animation(self, ax):
        ax.set_xlim(0 - 1, self.frames + 1)
        ax.tick_params(axis='x', labelsize=self.fontsize, width=2, length=8)
        ax.set_xlabel('Timesteps', fontsize=self.fontsize, labelpad=10)
        ax.set_ylim(self.episode_rewards.min() - 0.2, self.episode_rewards.max() + 0.2)
        ax.tick_params(axis='y', labelsize=self.fontsize, width=2, length=8)
        ax.set_ylabel('Reward', fontsize=self.fontsize, labelpad=0)

        # ax.set_title('Reward', fontsize=self.fontsize, pad=10)

        # ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # force integer ticks
        ax.grid(linestyle='--', linewidth=2, alpha=0.8)

        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # remove frame
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        lines = [ax.plot([], [], color='black', linewidth=2)[0]]
        return lines

    def _init_error_animation(self, ax):
        ax.set_xlim(0, self.frames)
        ax.set_ylim(self.episode_errors.min(), self.episode_errors.max())
        ax.set_xlabel('Timesteps', fontsize=self.fontsize)
        ax.set_ylabel('Prediction error', fontsize=self.fontsize)

        # remove frame
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        lines = [ax.plot([], [], color=self.prediction_colors[l])[0] for l in range(self.n_landmarks)]
        return lines

    def new_frame_seq(self):
        return iter(range(self.frames))

    def _init_draw(self):
        lines = self.lines_episode + self.lines_reward
        for l in lines:
            l.set_data([], [])


