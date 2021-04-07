"""

Determine the dh parameters of the robot given the coordinates of the axes and direction of normal
"""

import matplotlib.pyplot as plt
class manipulator:
    def __init__(self, base_axes, base_loc = [0,0,0]):
        self.axes = [base_axes]
        self.locs = [base_loc]

    def add_joint(self, joint_type, axis, loc):
        # currently only supports prismatic and revolute joints


    def find_dh_params(self):

    def plot_axis(self):
        pass

    def plot_manipulator(self):
        pass