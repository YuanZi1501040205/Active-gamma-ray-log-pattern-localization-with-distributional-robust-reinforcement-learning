import numpy as np
import cv2
from sacd.utils import iou
from sacd.utils import plot_state
from gym import spaces
import random
import os

class Env():
    def __init__(self, mode='train', num_actions=5,
                 observation_len=512, target_len=3000, target_sample_num=1, action_ratio=0.2, tau=0.5, step_max_time=25,
                 discount_ratio=1, monitor=False):
        # training is more strictly to ensure testing success
        if mode == 'train':
            tau = 0.6
        else:
            tau = 0.5
        self.tau = tau  # terminate threshold: iou > tau stop this trajectory, different from Agent's tau
        self.d_r = discount_ratio
        self.step_count = 0
        self.step_limitation = step_max_time
        self.action_ratio = action_ratio
        self.observation_len = observation_len
        self.mode = mode
        self.well = self.sample_well(mode)
        self.well_name = 'well01_01'
        self.info = 'empty'
        self.target_len = target_len
        self.target_sample_num = target_sample_num

        # choose the target fragment
        self.target_len = self.target_len
        target_samples = []
        for _ in range(self.target_sample_num):
            target_samples.append(int(self.target_len / 2) + (_ + 1)* (self.well.shape[1] - self.target_len)/(self.target_sample_num + 1))
        self.target_center = int(random.sample(target_samples, 1)[0])
        self.target_x_l = self.target_center - int(self.target_len / 2)
        self.target_x_r = self.target_center + int(self.target_len / 2) - 1
        self.target = self.well[1][self.target_x_l: self.target_x_r + 1]

        # environment boundary
        self.environment_left_edge = 0
        self.environment_right_edge = int(self.well.shape[1] - 1)

        # initial target label window position on the reference trace
        if self.well[2][self.target_x_l] < self.environment_left_edge:
            self.target_label_x_l = self.environment_left_edge
        else:
            self.target_label_x_l = self.well[2][self.target_x_l]
        if self.well[2][self.target_x_r] > self.environment_right_edge:
            self.target_label_x_r = self.environment_right_edge
        else:
            self.target_label_x_r = self.well[2][self.target_x_r]

        # action space state space
        self.action = np.zeros(5)
        self.action_space = spaces.Discrete(num_actions)
        self.observation = np.zeros((2, self.observation_len))
        self.observation_space = np.zeros((2, self.observation_len))

        # record the trajectory to video
        self.done = False # the flag which indicates if the trajectory is ended (find the target/ over the trail limitation time)
        self.monitor = monitor
        self.video = []

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]

    def update_target(self):
        # choose the target signal fragment
        self.target_len = self.target_len
        target_samples = []
        for _ in range(self.target_sample_num):
            target_samples.append(int(self.target_len / 2) + (_ + 1)* (self.well.shape[1] - self.target_len)/(self.target_sample_num + 1))
        self.target_center = int(random.sample(target_samples, 1)[0])
        self.target_x_l = self.target_center - int(self.target_len / 2)
        self.target_x_r = self.target_center + int(self.target_len / 2) - 1
        self.target = self.well[1][self.target_x_l: self.target_x_r + 1]

        # environment boundary
        self.environment_left_edge = 0
        self.environment_right_edge = int(self.well.shape[1] - 1)

        # initial target label window position on the reference trace
        if self.well[2][self.target_x_l] < self.environment_left_edge:
            self.target_label_x_l = self.environment_left_edge
        else:
            self.target_label_x_l = self.well[2][self.target_x_l]
        if self.well[2][self.target_x_r] > self.environment_right_edge:
            self.target_label_x_r = self.environment_right_edge
        else:
            self.target_label_x_r = self.well[2][self.target_x_r]

    def sample_well(self, mode):

        path = "./dataset/" + mode
        wells = []
        for x in os.listdir(path):
            if x.endswith('txt'):
                wells.append(x)
        selected_wells = random.sample(wells, k=1)
        print(selected_wells)
        filename = selected_wells[0].split('.')[0]

        well_name = filename
        self.well_name = well_name
        well_index = well_name.split('well')[-1]
        well = np.loadtxt('dataset/' + mode + '/well' + well_index + '.txt')# vector size of the whole signal is (9000,)
        return well

    def video_record(self):

        if not self.done:
            well = self.well
            target = [self.target_x_l, self.target_x_r]
            window = [self.window_x_l, self.window_x_r]
            step = self.step_count
            img = plot_state(self.well_name, well, target, window, step)
            self.video.append(img)
        else:
            if self.info == 'find' or self.info == 'over step but find':
                frameSize = (self.video[0].shape[1], self.video[0].shape[0])
                video_file = 'tmp/video/' + self.well_name
                out = cv2.VideoWriter(video_file + 'output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, frameSize, True)
                for i in range(len(self.video)):
                    img = self.video[i]
                    out.write(img)
                out.release()

    def reset(self):

        # set the window back to the global scope, target remain the same as the initial
        self.done = False # flag which indicate if the trajectory is end(find the target/ over the trail limitation time)
        self.step_count = 0

        # choose new well
        self.well = self.sample_well(self.mode)

        # sample target signal fragment
        self.update_target()
        self.info = 'empty'
        self.resized_window = cv2.resize(self.well[0], (1, self.observation_len), interpolation=cv2.INTER_LINEAR)
        self.resized_window = self.resized_window.squeeze(1)
        self.resized_target_fragment = cv2.resize(self.target, (1, self.observation_len), interpolation=cv2.INTER_LINEAR)
        self.resized_target_fragment = self.resized_target_fragment.squeeze(1)
        self.window_x_l = self.environment_left_edge
        self.window_x_r = self.environment_right_edge
        self.window_len = int(self.window_x_r - self.window_x_l + 1)
        self.window_center = int((self.window_x_l + self.window_x_r) / 2)
        observation = np.array([self.resized_window, self.resized_target_fragment])
        if self.monitor:
            self.video = []
            self.video_record()
        return observation

    def step(self, action):
        stop_choose_flag = False

        # store the window before update
        window_x_l_cache = self.window_x_l
        window_x_r_cache = self.window_x_r

        # update the window position
        # move left
        if action == 0:
            left_move_step = int(self.action_ratio * self.window_len)
            self.window_x_l = self.window_x_l - left_move_step
            self.window_x_r = self.window_x_l + self.window_len - 1

        # move right
        elif action == 1:
            right_move_step = int(self.action_ratio * self.window_len)
            self.window_x_r = self.window_x_r + right_move_step
            self.window_x_l = self.window_x_r - int(self.window_len - 1)

        # shrink the window
        elif action == 2:
            zoom_in_step = int((1 - self.action_ratio) * self.window_len / 2)
            self.window_x_r = self.window_center + zoom_in_step
            self.window_x_l = self.window_center - zoom_in_step

        # window expand
        elif action == 3:
            zoom_out_step = int(self.action_ratio * self.window_len / 2)
            # normal zoom out
            self.window_x_r = self.window_x_r + zoom_out_step
            self.window_x_l = self.window_x_l - zoom_out_step
        # stop action does not change the window
        else:
            stop_choose_flag = True

        # keep still if any over step happened
        if self.window_x_l >= self.window_x_r or self.window_x_l < self.environment_left_edge or self.window_x_r > self.environment_right_edge:
            self.window_x_r = window_x_r_cache
            self.window_x_l = window_x_l_cache

        # update window's center by using the x_l and x_r calculated before
        self.window_center = int((self.window_x_r + self.window_x_l) / 2)
        self.window_len = int(self.window_x_r - self.window_x_l + 1)

        # refresh the observation based on the new window
        resized_window = cv2.resize(self.well[0][self.window_x_l: self.window_x_r], (1, self.observation_len),
                                    interpolation=cv2.INTER_LINEAR)
        resized_window = resized_window.squeeze(1)
        resized_target_fragment = cv2.resize(self.target, (1, self.observation_len), interpolation=cv2.INTER_LINEAR)
        resized_target_fragment = resized_target_fragment.squeeze(1)
        observation_ = np.array([resized_window, resized_target_fragment])

        # reward
        # IoU of before update
        iou_bf = iou([window_x_l_cache, window_x_r_cache], [self.target_label_x_l, self.target_label_x_r])
        # IoU of after update
        iou_af = iou([self.window_x_l, self.window_x_r], [self.target_label_x_l, self.target_label_x_r])

        self.step_count = self.step_count + 1
        if stop_choose_flag and iou_af >= self.tau:
            reward = 10
            self.done = True
            self.info = 'find'
        elif stop_choose_flag and iou_af < self.tau:
            reward = -10
            self.done = True
            self.info = 'find false'
        else:
            if iou_af > iou_bf:
                reward = 1
            else:
                reward = -1
        reward = reward * self.d_r ** (self.step_count - 1)
        if self.step_count > self.step_limitation:
            if iou_af >= self.tau:
                self.info = 'over step but find'
            else:
                self.info = 'over step and not find'
            self.done = True

        done = self.done
        info = self.info

        if done:
            print('info:', info)
        if self.monitor:
            self.video_record()
        return observation_, reward, done, info
