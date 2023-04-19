import numpy as np


class HolisticSkeleton:

    def __init__(self):

        self.pose_landmark_num = 33
        self.dims = 4
     
        self.has_pose = True
        self.pose = np.array((self.pose_landmark_num, self.dims), dtype=np.float32)

        self.mp_results = None


    def update(self, mp_results):

        self.mp_results = mp_results
        self.pose[:, :] = mp_results.pose_landmarks

    def get_pose(self):
        return self.pose



