import cv2 as cv
from time import perf_counter
from skyeye.utils.opencv import Webcam, wait_key

import mediapipe as mp
import yaml

from holistic_data import HolisticData

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


    frame_width = 640
    frame_height = 480
    use_V4L2 = True
    autofocus = False
    auto_exposure = True

    # Set mediapipe using GPU
    with open(r'mediapipe.yaml', 'r', encoding='utf-8') as f:
        inputs = yaml.load(f, Loader=yaml.Loader)
    enable_gpu = inputs['enable_gpu']

    webcam = Webcam()
    if webcam.is_open():
        webcam.release()

    device = 0 # Device ID
    webcam.open(device, width=frame_width, height=frame_height,
        use_V4L2=use_V4L2, autofocus=autofocus, auto_exposure=auto_exposure)

    holistic_data = HolisticData()


    frame_count = 0
    while True:

        frame_count += 1
        print("frame_count = {}".format(frame_count))
        time_start = perf_counter()

        frame = webcam.read()
        if frame is None:
            break

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


    # cleanup the camera and close any open windows
    if webcam.is_open():
        webcam.release()

    cv.destroyAllWindows()
