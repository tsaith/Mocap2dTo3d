import numpy as np


class BaselinePose:

    def __init__(self):

        self.num_pose_landmarks = 17
        self.dims = 3
     
        self.has_pose = True
        self.pose = np.zeros((self.num_pose_landmarks, self.dims), dtype=np.float32)

        self.index_hip = 0
        self.index_right_hip = 1
        self.index_right_knee = 2
        self.index_right_ankle = 3
        self.index_left_hip = 4
        self.index_left_knee = 5
        self.index_left_ankle = 6
        self.index_spine02 = 7
        self.index_neck01 = 8
        self.index_nose = 9
        self.index_head_top = 10
        self.index_left_shoulder = 11
        self.index_left_elbow = 12
        self.index_left_wrist = 13
        self.index_right_shoulder = 14
        self.index_right_elbow = 15
        self.index_right_wrist = 16

        self.holistic_data = None 

    def update(self, holistic_data):

        self.holistic_data = holistic_data 

        # Pose
        self.update_pose()


    def update_pose(self):

        holistic_pose = self.holistic_data.pose

        i = self.index_hip
        self.pose[i, :] = get_point(holistic_pose, index)


    def get_point(self, data, index):

        point = np.zeros(self.dims, dtype=np.float32)
        point[:] = data[index, 0:self.dims]

        return point
    
    def get_pose(self):
        return self.pose



