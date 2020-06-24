import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

from pose3dmodules import *
from posesocket import PoseTransmitter
from posescheduler import PoseScheduler
from poseinferscheduler import PoseInferScheduler

import time


def run_3dpose(net):
    cpu = False
    downscale_resolution = True
    transmit_over_socket = False

    # NOTE: prerecorded videos not included
    useprerecorded = True
    waitforready = True

    net = net.eval()
    net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    r_previous_poses = []
    l_previous_poses = []
    delay = 33

    # the distance between the centers of the camera lenses
    camera_dist = 0.16
    # the separation distance in pixels of the stereo cameras
    camera_pixel_dev = 50

    # hard coded for now, captured using calculate_camera_intrinsics.py
    intr_640x480 = [
        [515.99751233,  0.0,            297.71434208],
        [0.0,           516.03304654,   249.19311156],
        [0.0,           0.0,            1.0         ]
    ]

    intr_360x270 = [
        [289.87290864,      0.0,            177.24868],
        [0.0,               289.83381563,   133.56638493],
        [0.0,               0.0,            1.0         ]
    ]


    if not useprerecorded:
        # NOTE: Set the camera ids to match your setup
        Lcam = cv2.VideoCapture(2)
        Rcam = cv2.VideoCapture(4)
    else:
        camera_dist = 0.15
        # NOTE: these videos are not included
        # Lcam = cv2.VideoCapture("testvideo/L_video_50cm.avi")
        # Rcam = cv2.VideoCapture("testvideo/R_video_50cm.avi")
        Lcam = cv2.VideoCapture("testvideo/L_video_multi_15cm.avi")
        Rcam = cv2.VideoCapture("testvideo/R_video_multi_15cm.avi")

        # due to unfortunate offset in the frames
        for i in range(4):
            Lcam.read()


    pose_extraction_scheduler = PoseScheduler()
    pose_infer_scheduler = PoseInferScheduler(net, stride, upsample_ratio, cpu)
    pose_infer_scheduler.start_infer()

    pose3d = None
    Rimg = None
    Limg = None
    newpose, r_current_poses, l_current_poses, Rimg_synced, Limg_synced = False, None, None, None, None

    if transmit_over_socket:
        posetransmitter = PoseTransmitter(host="127.0.0.1", port=1234)
        posetransmitter.await_connection()

    canadvance = True
    newframetime = time.time()

    previous_pose_transmit_time = time.time()

    while True:
        begintime = time.time()

        if not waitforready or (waitforready and canadvance):
            # if (time.time() - newframetime)*1000 > 5:
            #     print("fps:", int(1 / (time.time() - newframetime)), " -  frametime:", int((time.time() - newframetime)*1000), "ms")
            newframetime = time.time()
            
            Limg = Lcam.read()[1]
            Rimg = Rcam.read()[1]

            if Rimg is None:
                Rcam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                Rimg = Rcam.read()[1]

            if Limg is None:
                Lcam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                Limg = Lcam.read()[1]

            if downscale_resolution:
                Rimg = cv2.resize(Rimg, (360, 270))
                Limg = cv2.resize(Limg, (360, 270))

            pose_infer_scheduler.set_images(Rimg, Limg)

            Lnewpose, r_pose_data, Rimg_sync = pose_infer_scheduler.get_left_pose()
            Rnewpose, l_pose_data, Limg_sync = pose_infer_scheduler.get_right_pose()
            if r_pose_data is None or l_pose_data is None:
                continue

            Rheatmaps, Rpafs, Rscale, Rpad = r_pose_data
            Lheatmaps, Lpafs, Lscale, Lpad = l_pose_data

            if pose_extraction_scheduler.is_done():
                pose_extraction_scheduler.schedule_new_stereo_extract("left", Rimg_sync, r_previous_poses, Rheatmaps, Rpafs, Rscale, Rpad, num_keypoints, stride, upsample_ratio)
                pose_extraction_scheduler.schedule_new_stereo_extract("right", Limg_sync, l_previous_poses, Lheatmaps, Lpafs, Lscale, Lpad, num_keypoints, stride, upsample_ratio)

                newpose, r_current_poses, l_current_poses, Rimg_synced, Limg_synced = pose_extraction_scheduler.execute_schedule()

            canadvance = False

        if newpose:
            # force the feed to run at most 30 fps, especially useful on more powerful devices
            # while time.time() < previous_pose_transmit_time + 1 / 30:
            #     continue

            previous_pose_transmit_time = time.time()

            current_pose_list_l = poses_to_list(l_current_poses)
            current_pose_list_r = poses_to_list(r_current_poses)
            previous_pose_list_l = poses_to_list(l_previous_poses)
            previous_pose_list_r = poses_to_list(r_previous_poses)

            current_pose_list_l = smooth_2d_poses(current_pose_list_l, previous_pose_list_l)
            current_pose_list_r = smooth_2d_poses(current_pose_list_r, previous_pose_list_r)

            canadvance = True
            r_previous_poses = apply_to_previous(r_previous_poses, r_current_poses)
            l_previous_poses = apply_to_previous(l_previous_poses, l_current_poses)

            height, width, channels = Rimg.shape
            pose3d = None
            try:
                height, width, _ = Limg.shape
                if height == 480 and width == 640:
                    pose3d = pose_make_3d(current_pose_list_r, current_pose_list_l, intr_640x480, camera_dist, width, camera_pixel_dev)
                elif height == 270 and width == 360:
                    pose3d = pose_make_3d(current_pose_list_r, current_pose_list_l, intr_360x270, camera_dist, width, camera_pixel_dev)
                else:
                    print("incorrect image shape of", width, height)

            except:
                continue

            if transmit_over_socket:
                posetransmitter.transmit_pose(pose3d)


        if len(r_previous_poses) == 0 or len(l_previous_poses) == 0:
            continue

        Rimg = draw_pose(r_current_poses, Rimg_synced)
        Limg = draw_pose(l_current_poses, Limg_synced)
        
        cv2.imshow('img', np.hstack([Rimg, Limg]))
        key = cv2.waitKey(1)
        if key == ord('q'):
            pose_infer_scheduler.stop_infer()
            return

        # print("fps:", int(1 / (time.time() - begintime)))



if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)

    run_3dpose(net)
