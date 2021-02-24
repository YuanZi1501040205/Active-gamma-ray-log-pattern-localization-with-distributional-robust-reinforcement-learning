from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_state(well_name, well, target, window, step):
    """well: [3, 9000] stores reference and target signal vectors and matching labels
       target: [target  x left, target x right] target signal fragment
       window: [window x left, window x right] current observation bbox"""

    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号


    gt_window = [well[2][target[0]], well[2][target[1]]]
    x_axis = np.arange(well.shape[1])
    plt.figure(figsize=(10, 5))

    ax1 = plt.subplot(211)
    title = 'step ' + str(step)
    plt.title(title, fontsize=13, fontweight='bold')
    gt_rect = plt.Rectangle((gt_window[0], min(well[0])), int(gt_window[1] - gt_window[0]), max(well[0]), fill=True,
                            ls='--', alpha=0.6, color='#00FF00', lw=2)
    ax1.add_patch(gt_rect)
    gt_rect.set_label('ground truth')

    roi_rect = plt.Rectangle((window[0], min(well[0]) - 0.05), int(window[1] - window[0]), max(well[0]) + 0.1,
                             fill=False, color='k', lw=1.5)
    ax1.add_patch(roi_rect)
    roi_rect.set_label('observation')

    plt.plot(x_axis, well[0], label='signal', color="blue", linewidth=1.5)

    # bbx
    bbx_xy = roi_rect.get_xy()
    bbx_width = roi_rect.get_width()

    # gt bbx
    gt_bbx_xy = gt_rect.get_xy()
    gt_bbx_width = gt_rect.get_width()

    plt.xticks(fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')

    plt.ylabel('Amplitude', fontsize=13, fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    ax2 = plt.subplot(212, sharex=ax1)

    target_rect = plt.Rectangle((target[0], min(well[1])), int(target[1] - target[0]), max(well[1]), fill=False,
                                color='r', lw=1.5)
    ax2.add_patch(target_rect)

    plt.plot(x_axis, well[1], label='distortion of signal', color="black", linewidth=1.5)

    target_rect.set_label('target')

    plt.xticks(fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')

    plt.ylabel('Amplitude', fontsize=13, fontweight='bold')
    plt.xlabel('Depth', fontsize=13, fontweight='bold')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    figure_file = './logs/'  + well_name + '_' + title
    plt.savefig(figure_file + '.png')
    plt.close('all')
    plt.cla()
    plt.clf()

    img = cv2.imread(figure_file + '.png')
    os.remove(figure_file + '.png')

    plot_bbx(bbx_xy, bbx_width, x_axis, well, gt_bbx_xy, gt_bbx_width, well_name, title)

    return img

def plot_bbx(bbx_xy, bbx_width, x_axis, well, gt_bbx_xy, gt_bbx_width, well_name, title):
    # plot bbx
    plt.style.use('dark_background')
    plt.figure(figsize=(5, 5))

    bbx_l = int(bbx_xy[0])
    bbx_r = int(bbx_l + bbx_width)

    plt.plot(x_axis[bbx_l: bbx_r], well[0][bbx_l: bbx_r], color='dodgerblue', linewidth=1.5)
    gt_bbx_l = max(int(gt_bbx_xy[0]), bbx_l)
    gt_bbx_r = min(int(gt_bbx_width + int(gt_bbx_xy[0])), bbx_r)
    if (gt_bbx_l >= bbx_l and gt_bbx_l <= bbx_r) or (gt_bbx_r >= bbx_l and gt_bbx_r <= bbx_r):
        plt.plot(x_axis[gt_bbx_l: gt_bbx_r], well[0][gt_bbx_l: gt_bbx_r], color='greenyellow', linewidth=1.5)

    plt.axis('off')
    figure_file = './tmp/' + well_name + '_' + title
    plt.savefig(figure_file + '.eps', format='eps', dpi=300)
    plt.close('all')
    plt.cla()
    plt.clf()

def iou(window1, window2):
    """intersection of window1 and window2 / union of window 1 and window 2
    window shape is [x_left, x_right]"""
    # determine the intersection window
    in_window_x_l = max(window1[0], window2[0])
    in_window_x_r = min(window1[1], window2[1])
    inter_area = max((in_window_x_r - in_window_x_l), 0)
    if inter_area == 0:
        return 0
    window1_area = abs(window1[1] - window1[0])
    window2_area = abs(window2[1] - window2[0])
    iou_ratio = inter_area / (window1_area + window2_area - inter_area)
    return iou_ratio

def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
