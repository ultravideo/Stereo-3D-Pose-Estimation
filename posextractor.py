import cv2
import numpy as np
import torch
import copy
import time
import math
from operator import itemgetter
import threading

from models.with_mobilenet import PoseEstimationWithMobileNet
# from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose
from val import normalize, pad_width

class PoseExtractor (threading.Thread):
    def __init__(self, previous_pose, current_pose_target, heatmaps, pafs, scale, pad, num_keypoints, stride, upsample_ratio):
        threading.Thread.__init__(self)

        self.BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
        self.BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])

        self.previous_pose = previous_pose
        self.heatmaps = heatmaps
        self.pafs = pafs
        self.scale = scale
        self.pad = pad
        self.num_keypoints = num_keypoints
        self.stride = stride
        self.upsample_ratio = upsample_ratio
        self.current_pose_target = current_pose_target

        self.current_poses = None

    def run(self):
        self.extract_pose(self.heatmaps, self.pafs, self.scale, self.pad, self.num_keypoints, self.stride, self.upsample_ratio)
        self.track_poses(self.previous_pose, self.current_poses, smooth=True)

    def get_pose(self):
        return self.current_poses

    def linspace2d(self, start, stop, n=10):
        points = 1 / (n - 1) * (stop - start)
        return points[:, None] * np.arange(n) + start[:, None]

    def extract_pose(self, heatmaps, pafs, scale, pad, num_keypoints, stride, upsample_ratio):
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += self.extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = self.group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        self.current_poses = []

        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            self.current_poses.append(pose)

        # return current_poses

    def extract_keypoints(self, heatmap, all_keypoints, total_keypoint_num):
        heatmap[heatmap < 0.1] = 0
        heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
        heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
        heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
        heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
        heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
        heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

        heatmap_peaks = (heatmap_center > heatmap_left) &\
                        (heatmap_center > heatmap_right) &\
                        (heatmap_center > heatmap_up) &\
                        (heatmap_center > heatmap_down)
        heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
        keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
        keypoints = sorted(keypoints, key=itemgetter(0))

        suppressed = np.zeros(len(keypoints), np.uint8)
        keypoints_with_score_and_id = []
        keypoint_num = 0
        for i in range(len(keypoints)):
            if suppressed[i]:
                continue
            for j in range(i+1, len(keypoints)):
                if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                            (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                    suppressed[j] = 1
            keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],
                                        total_keypoint_num + keypoint_num)
            keypoints_with_score_and_id.append(keypoint_with_score_and_id)
            keypoint_num += 1
        all_keypoints.append(keypoints_with_score_and_id)
        return keypoint_num

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05, demo=False):
        pose_entries = []
        all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
        for part_id in range(len(self.BODY_PARTS_PAF_IDS)):
            part_pafs = pafs[:, :, self.BODY_PARTS_PAF_IDS[part_id]]
            kpts_a = all_keypoints_by_type[self.BODY_PARTS_KPT_IDS[part_id][0]]
            kpts_b = all_keypoints_by_type[self.BODY_PARTS_KPT_IDS[part_id][1]]
            num_kpts_a = len(kpts_a)
            num_kpts_b = len(kpts_b)
            kpt_a_id = self.BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = self.BODY_PARTS_KPT_IDS[part_id][1]

            if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
                continue
            elif num_kpts_a == 0:  # body part has just 'b' keypoints
                for i in range(num_kpts_b):
                    num = 0
                    for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                        if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                            num += 1
                            continue
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                        pose_entry[-1] = 1                   # num keypoints in pose
                        pose_entry[-2] = kpts_b[i][2]        # pose score
                        pose_entries.append(pose_entry)
                continue
            elif num_kpts_b == 0:  # body part has just 'a' keypoints
                for i in range(num_kpts_a):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                            num += 1
                            continue
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = kpts_a[i][3]
                        pose_entry[-1] = 1
                        pose_entry[-2] = kpts_a[i][2]
                        pose_entries.append(pose_entry)
                continue

            connections = []
            for i in range(num_kpts_a):
                kpt_a = np.array(kpts_a[i][0:2])
                for j in range(num_kpts_b):
                    kpt_b = np.array(kpts_b[j][0:2])
                    mid_point = [(), ()]
                    mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                                    int(round((kpt_a[1] + kpt_b[1]) * 0.5)))
                    mid_point[1] = mid_point[0]

                    vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                    vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                    if vec_norm == 0:
                        continue
                    vec[0] /= vec_norm
                    vec[1] /= vec_norm
                    cur_point_score = (vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0] +
                                    vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1])

                    height_n = pafs.shape[0] // 2
                    success_ratio = 0
                    point_num = 10  # number of points to integration over paf
                    if cur_point_score > -100:
                        passed_point_score = 0
                        passed_point_num = 0
                        x, y = self.linspace2d(kpt_a, kpt_b)
                        for point_idx in range(point_num):
                            if not demo:
                                px = int(round(x[point_idx]))
                                py = int(round(y[point_idx]))
                            else:
                                px = int(x[point_idx])
                                py = int(y[point_idx])
                            paf = part_pafs[py, px, 0:2]
                            cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                            if cur_point_score > min_paf_score:
                                passed_point_score += cur_point_score
                                passed_point_num += 1
                        success_ratio = passed_point_num / point_num
                        ratio = 0
                        if passed_point_num > 0:
                            ratio = passed_point_score / passed_point_num
                        ratio += min(height_n / vec_norm - 1, 0)
                    if ratio > 0 and success_ratio > 0.8:
                        score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                        connections.append([i, j, ratio, score_all])
            if len(connections) > 0:
                connections = sorted(connections, key=itemgetter(2), reverse=True)

            num_connections = min(num_kpts_a, num_kpts_b)
            has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
            has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
            filtered_connections = []
            for row in range(len(connections)):
                if len(filtered_connections) == num_connections:
                    break
                i, j, cur_point_score = connections[row][0:3]
                if not has_kpt_a[i] and not has_kpt_b[j]:
                    filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                    has_kpt_a[i] = 1
                    has_kpt_b[j] = 1
            connections = filtered_connections
            if len(connections) == 0:
                continue

            if part_id == 0:
                pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
                for i in range(len(connections)):
                    pose_entries[i][self.BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                    pose_entries[i][self.BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                    pose_entries[i][-1] = 2
                    pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
            elif part_id == 17 or part_id == 18:
                kpt_a_id = self.BODY_PARTS_KPT_IDS[part_id][0]
                kpt_b_id = self.BODY_PARTS_KPT_IDS[part_id][1]
                for i in range(len(connections)):
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                        elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                            pose_entries[j][kpt_a_id] = connections[i][0]
                continue
            else:
                kpt_a_id = self.BODY_PARTS_KPT_IDS[part_id][0]
                kpt_b_id = self.BODY_PARTS_KPT_IDS[part_id][1]
                for i in range(len(connections)):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0]:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                            num += 1
                            pose_entries[j][-1] += 1
                            pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = connections[i][0]
                        pose_entry[kpt_b_id] = connections[i][1]
                        pose_entry[-1] = 2
                        pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                        pose_entries.append(pose_entry)

        filtered_entries = []
        for i in range(len(pose_entries)):
            if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
                continue
            filtered_entries.append(pose_entries[i])
        pose_entries = np.asarray(filtered_entries)
        return pose_entries, all_keypoints

    def get_similarity(self, a, b, threshold=0.5):
        num_similar_kpt = 0
        for kpt_id in range(Pose.num_kpts):
            if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
                distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
                area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
                similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
                if similarity > threshold:
                    num_similar_kpt += 1
        return num_similar_kpt
    
    def track_poses(self, previous_poses, current_poses, threshold=3, smooth=False):
        """Propagate poses ids from previous frame results. Id is propagated,
        if there are at least `threshold` similar keypoints between pose from previous frame and current.
        If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

        :param previous_poses: poses from previous frame with ids
        :param current_poses: poses from current frame to assign ids
        :param threshold: minimal number of similar keypoints between poses
        :param smooth: smooth pose keypoints between frames
        :return: None
        """
        current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
        mask = np.ones(len(previous_poses), dtype=np.int32)
        for current_pose in current_poses:
            best_matched_id = None
            best_matched_pose_id = None
            best_matched_iou = 0
            for id, previous_pose in enumerate(previous_poses):
                if not mask[id]:
                    continue
                iou = self.get_similarity(current_pose, previous_pose)
                if iou > best_matched_iou:
                    best_matched_iou = iou
                    best_matched_pose_id = previous_pose.id
                    best_matched_id = id
            if best_matched_iou >= threshold:
                mask[best_matched_id] = 0
            else:  # pose not similar to any previous
                best_matched_pose_id = None
            current_pose.update_id(best_matched_pose_id)

            if smooth:
                for kpt_id in range(Pose.num_kpts):
                    if current_pose.keypoints[kpt_id, 0] == -1:
                        continue
                    # reuse filter if previous pose has valid filter
                    if (best_matched_pose_id is not None
                            and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                        current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                    current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                    current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
                current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
