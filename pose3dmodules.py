
import cv2
import numpy as np
import torch
import copy
import time

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

# nose, neck, r_sho, r_elb, r_wri, l_sho, l_elb, l_wri, r_hip, l_hip, r_eye, l_eye, r_ear, l_ear]
upperbody_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15, 16, 17]

upperbody_keypoints_d = {
    "nose" : 0,
    "neck" : 1,
    "r_sho" : 2,
    "r_elb" : 3,
    "r_wri" : 4,
    "l_sho" : 5,
    "l_elb" : 6,
    "l_wri" : 7,
    "r_hip" : 8,
    "l_hip" : 11,
    "r_eye" : 14,
    "l_eye" : 15,
    "r_ear" : 16,
    "l_ear" : 17 }

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def project_pixel_to_world(pixel, depth, intr):
    flip_transform = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    cx = intr[0][2]
    cy = intr[1][2]
    fx = intr[0][0]
    fy = intr[1][1]
    x = (pixel[0] - cx) * depth / fx
    y = (pixel[1] - cy) * depth / fy
    z = depth

    return np.dot(np.asarray([x, y, z]), flip_transform)


def depth_from_disparity_parallax(lpoint, rpoint, intr, camera_dist, img_width):
    lx = lpoint[0]
    rx = rpoint[0]
    depth = abs(rx - lx)
    if depth <= 1:
        print("invalid depth of", depth, rpoint, lpoint)

    depth = (camera_dist * intr[0][0]) / (depth * img_width)
    world_point = project_pixel_to_world([lpoint[0], lpoint[1]], depth, intr)
    return np.asarray(world_point) * 100.0

def make_poses_valid(poses_l, poses_r, camera_pixel_dev):
    return

def smooth_2d_poses(poses, previous_poses):
    invalid = [0, -1]
    if len(previous_poses) == 0 or previous_poses is None:
        return poses

    if len(poses) is not len(previous_poses):
        poses = [poses[0]]
        previous_poses = [previous_poses[0]]

    for p in range(len(poses)):
        current_pose = np.asarray(poses[p], np.float32)
        current_previous_pose = np.asarray(previous_poses[p], np.float32)

        for i in range(len(current_pose)):
            # ipose = copy.deepcopy(current_pose[i])
            if current_pose[i][0] not in invalid and current_pose[i][1] not in invalid:
                current_pose[i][0] = current_pose[i][0]*0.75 + current_previous_pose[i][0]*0.25
                current_pose[i][1] = current_pose[i][1]*0.75 + current_previous_pose[i][1]*0.25
                # print(current_pose[i], ipose, current_previous_pose[i])
            else:
                current_pose[i][0] = current_previous_pose[i][0]
                current_pose[i][1] = current_previous_pose[i][1]

        poses[p] = current_pose
    
    return poses

def poses_to_list(poses):
    poselist = []

    for p in poses:
        poselist.append(p.keypoints)

    return poselist

def remove_invalid_poses(poses_l, poses_r, camera_pixel_dev):
    return poses_l, poses_r


def pose_make_3d(poses_l, poses_r, intr, camera_dist, img_width, camera_pixel_dev):
    poses3d = []
    print(len(poses_l), len(poses_r))

    poses_l, poses_r = remove_invalid_poses(poses_l, poses_r, camera_pixel_dev)

    zeropoint = [0, 0, 0]

    for i in range(len(poses_l)):
        keypoints_l = copy.deepcopy(poses_l[i])
        keypoints_r = copy.deepcopy(poses_r[i])

        current_pose = []

        for kp in range(len(keypoints_l)):
            point_l = keypoints_l[kp]
            point_r = keypoints_r[kp]

            if (int(point_l[0]) == -1 and int(point_l[1]) == -1) or (int(point_r[0]) == -1 and int(point_r[1]) == -1):
                poses3d.append(zeropoint)
                print("invalid value of zero!")
                continue

            cpoint = depth_from_disparity_parallax(point_l, point_r, intr, camera_dist, img_width)
            current_pose.append(cpoint)
    
        poses3d.append(current_pose)
    
    return [poses3d]

def draw_pose(current_poses, img):
    orig_img = img
    for pose in current_poses:
        pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

    # for pose in current_poses:
        # cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
        #                 (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))

        # cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    return img

def extract_pose(heatmaps, pafs, scale, pad, num_keypoints, stride, upsample_ratio):
    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []

    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    return current_poses


def apply_to_previous(previous, current):
    if len(previous) is 0 or len(current) is 0:
        return current
    
    p = 0
    for p in range(min(len(previous), len(current))):

        previouskeypoints = previous[p].keypoints
        currentkeypoints = current[p].keypoints

        for i in range(len(currentkeypoints)):
            if currentkeypoints[i][0] == -1:
                continue
            else:
                previouskeypoints[i] = currentkeypoints[i]

        previous[p].keypoints = previouskeypoints

    if len(current) > len(previous):
        previous.extend(current[p:])

    return previous
