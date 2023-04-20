import numpy as np


class BaselineData:

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

        self.bl_inputs = None

    def update(self, holistic_data):

        self.holistic_data = holistic_data 
        data = holistic_data

        left_ear = self.get_point(data.get_left_ear())
        right_ear = self.get_point(data.get_right_ear())
        head = 0.5*(left_ear + right_ear)

        # Hip
        left_hip = self.get_point(data.get_left_hip())
        right_hip = self.get_point(data.get_right_hip())
        hip = 0.5*(left_hip + right_hip)

        self.pose[self.index_hip, :] = hip

        # Right hip
        self.pose[self.index_right_hip, :] = right_hip

        # Right knee
        right_knee = self.get_point(data.get_right_knee())

        self.pose[self.index_right_knee, :] = right_knee

        # Right ankle
        right_ankle = self.get_point(data.get_right_ankle())

        self.pose[self.index_right_ankle, :] = right_ankle

        # Left hip
        self.pose[self.index_left_hip, :] = left_hip

        # Left knee
        left_knee = self.get_point(data.get_left_knee())

        self.pose[self.index_left_knee, :] = left_knee

        # Left ankle
        left_ankle = self.get_point(data.get_left_ankle())

        self.pose[self.index_left_ankle, :] = left_ankle

        # Spine02
        left_shoulder = self.get_point(data.get_left_shoulder())
        right_shoulder = self.get_point(data.get_right_shoulder())
        shoulder_center = 0.5*(left_shoulder + right_shoulder)
        spine02 = 0.5*(hip + shoulder_center)

        self.pose[self.index_left_ankle, :] = spine02

        # Neck01
        neck01 = 0.5*(head + shoulder_center)
        self.pose[self.index_neck01, :] = neck01

        # Nose
        nose = self.get_point(data.get_nose())
        self.pose[self.index_nose, :] = nose

        # Head top
        head_top = head 
        self.pose[self.index_head_top, :] = head_top

        # Left shoulder
        left_shoulder = self.get_point(data.get_left_shoulder())
        self.pose[self.index_left_shoulder, :] = left_shoulder

        # Left elbow
        left_elbow = self.get_point(data.get_left_elbow())
        self.pose[self.index_left_elbow, :] = left_elbow

        # Left wrist
        left_wrist = self.get_point(data.get_left_wrist())
        self.pose[self.index_left_wrist, :] = left_wrist

        # right shoulder
        right_shoulder = self.get_point(data.get_right_shoulder())
        self.pose[self.index_right_shoulder, :] = right_shoulder

        # right elbow
        right_elbow = self.get_point(data.get_right_elbow())
        self.pose[self.index_right_elbow, :] = right_elbow

        # right wrist
        right_wrist = self.get_point(data.get_right_wrist())
        self.pose[self.index_right_wrist, :] = right_wrist


    def get_point(self, data):

        point = np.zeros(self.dims, dtype=np.float32)
        point[:] = data[0:self.dims]

        return point
    
    def get_pose(self):
        return self.pose


    def get_pose_flattened(self):
        return self.pose.flatten() 
    
    def get_bl_inputs(self):

        num_points = self.num_pose_landmarks - 1
        self.bl_inputs = np.zeros([1, num_points*2], dtype=np.float32)

        for i in range(num_points):

            j = i+1

            x = self.pose[j, 0]    
            y = self.pose[j, 1]    

            self.bl_inputs[0, 2*i] = x
            self.bl_inputs[0, 2*i+1] = x

        return self.bl_inputs
          
    def normalize_inputs(self):
        pass

    def unnormalize_outputs(self, data):

        hip_point = np.zeros([1, 3], dtype=np.float32)

        out = np.concatenate([hip_point, data], axis=1)

        return out