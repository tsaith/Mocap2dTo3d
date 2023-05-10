import cv2 as cv
from time import perf_counter
from skyeye.utils.opencv import Webcam, wait_key

import mediapipe as mp
import yaml

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.model import LinearModel, weight_init
from holistic_data import HolisticData
from baseline_data import BaselineData

from plot_utils import plot_bl_pose_2d, plot_bl_pose_3d

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic_detector = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)



def draw_msg(image, msg, x, y, y_shift=20, color=(0, 255,0)):

    cv.putText(image, msg, (x, y),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    y += y_shift

    return (x, y)


if __name__ == '__main__':

    use_webcam = False # True: use video; False: use webcam

    video_path = "videos/WalkAround.mp4"
    webcam_device = 0 # Device ID

    frame_width = 640
    frame_height = 480

    use_V4L2 = True
    autofocus = False
    auto_exposure = True

    baseline_checkpoint_path = "ckpt_best.pth.tar"

    # Set mediapipe using GPU
    with open(r'mediapipe.yaml', 'r', encoding='utf-8') as f:
        inputs = yaml.load(f, Loader=yaml.Loader)

    enable_gpu = inputs['enable_gpu']

    if use_webcam:

        webcam = Webcam()
        if webcam.is_open():
            webcam.release()

        cap = webcam.open(webcam_device, width=frame_width, height=frame_height,
            use_V4L2=use_V4L2, autofocus=autofocus, auto_exposure=auto_exposure)

    else:

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file. {video_path}")
            exit()


    holistic_data = HolisticData()
    baseline_data = BaselineData()


    # Baseline model
    model = LinearModel()
    model = model.cuda()
    model.apply(weight_init)

    # Load checkpoint 
    ckpt = torch.load(baseline_checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    stat_3d_path = "data/stat_3d.pth.tar"
    stat_3d = torch.load(stat_3d_path)
    baseline_data.set_stat_3d(stat_3d)


    frame_count = 0
    while True:

        frame_count += 1
        print("frame_count = {}".format(frame_count))
        time_start = perf_counter()

        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame  
        frame = cv.resize(frame, (frame_width, frame_height), interpolation=cv.INTER_AREA)


        frame = cv.flip(frame, 1)
        image = frame.copy()
        image_out = frame.copy()

        # Flip the image horizontally for a later selfie-view display, and convert
        if enable_gpu == 1:
            # the BGR image to RGBA.
            image = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
        else:
            # the BGR image to RGB.
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic_detector.process(image)

        holistic_data.update(results)
        baseline_data.update(holistic_data)

        inputs = baseline_data.get_bl_inputs()
        print(f"inputs shape: {inputs.shape}")

        print(f"pose: {baseline_data.pose}")
         
        inputs = torch.from_numpy(inputs)
        inputs = Variable(inputs.cuda())
        outputs = model(inputs)

        outputs = outputs.data.cpu().numpy()
        print(f"outputs shape: {outputs.shape}")

        outputs = baseline_data.add_hip(outputs)
        baseline_data.update_with_bl_pose_3d(outputs[0])
        pose = baseline_data.unnormalize()
        baseline_data.update_with_pose(pose)

        '''
        pose_2d_fig = plot_bl_pose_2d(bl_output, title="baseline pose 2d")
        pose_2d_fig.savefig("pose_2d.jpg")

        pose_3d_fig = plot_bl_pose_3d(bl_output, title="baseline pose 3d")
        pose_3d_fig.savefig("pose_3d.jpg")
        '''


        # Draw landmark annotation on the image.
        #mp_drawing.draw_landmarks(
        #    image_out, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)


        mp_drawing.draw_landmarks(
            image_out,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

        mp_drawing.draw_landmarks(
            image_out, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image_out, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image_out, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        #mp_drawing.plot_landmarks(results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Frame rate
        time_end = perf_counter()
        time_duration = time_end - time_start
        fps = int(1.0/time_duration)

        text_x = 20
        text_y = 20
        text_y_shift = 20

        msg = "fps: {}".format(fps)
        (text_x, text_y) = draw_msg(image_out, msg, text_x, text_y)

        # show the frame and record if the user presses a key
        cv.imshow("Win", image_out)

        # Exit while 'q' or 'Esc' is pressed
        key = wait_key(1)
        if key == ord("q") or key == 27: break


    if use_webcam:

        # cleanup the camera and close any open windows
        if webcam.is_open():
            webcam.release()
    else:

        if cap.isOpened():
            cap.release()        

    cv.destroyAllWindows()
