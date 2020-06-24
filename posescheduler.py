import time
import copy
import threading
from posextractor import PoseExtractor
from modules.pose import Pose


class PoseScheduler:
    def __init__(self, extractor_count=2):
        self.posextractors = []
        self.extractor_count = extractor_count

        self.left_thread = None
        self.right_thread = None

        self.lpose = None
        self.rpose = None

        self.l_img = None
        self.r_img = None

    def execute_schedule(self):
        if self.left_thread is not None and self.right_thread is not None:
            self.left_thread.start()
            self.right_thread.start()
            
            self.left_thread.join()
            self.right_thread.join()
            
            self.lpose = self.left_thread.get_pose()
            self.rpose = self.right_thread.get_pose()

            return True, self.lpose, self.rpose, self.l_img, self.r_img
        
        return False, self.lpose, self.rpose, self.l_img, self.r_img

    def schedule_new_stereo_extract(self, leftright, img, previous_pose, heatmaps, pafs, scale, pad, num_keypoints, stride, upsample_ratio):
        current_pose = None
        cthread = PoseExtractor(previous_pose, current_pose, heatmaps, pafs, scale, pad, num_keypoints, stride, upsample_ratio)

        if leftright is "left":
            self.left_thread = cthread
            self.l_img = copy.copy(img)
        elif leftright is "right":
            self.right_thread = cthread
            self.r_img = copy.copy(img)
        else:
            print("Invalid thread handedness:", leftright)

    def is_done(self):
        if self.left_thread is not None and self.right_thread is not None:
            if not self.left_thread.isAlive() and not self.right_thread.isAlive():
                return True
            else:
                time.sleep(0.001)
                return False
        else:
            return True

