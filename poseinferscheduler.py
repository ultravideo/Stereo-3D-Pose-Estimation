import threading
import time
import copy

from pose3dmodules import *


class PoseInferScheduler:
    def __init__(self, net, stride, upsample_ratio, cpu):
        self.net = net
        self.height = 0

        self.stride = stride
        self.upsample_ratio = upsample_ratio
        self.cpu = cpu

        self.left_img = None
        self.right_img = None

        self.exit_flag = False
        self.infer_thread = None
        self.img_lock = threading.Lock()
        self.pose_lock = threading.Lock()

        self.l_pose = None
        self.r_pose = None
        self.l_pose_img = None
        self.r_pose_img = None

        self.r_newpose = False
        self.l_newpose = False

    def stop_infer(self):
        self.exit_flag = True
        self.infer_thread.join()
        print("stopping infering thread")

    def start_infer(self):
        self.exit_flag = False
        self.infer_thread = threading.Thread(target=self.infer_pose_loop)
        self.infer_thread.start()
        print("infering thread started")

    def infer_pose_loop(self):
        while True:
            if self.exit_flag:
                break
            
            if self.left_img is None or self.right_img is None:
                time.sleep(1)
                continue

            self.img_lock.acquire()
            l_img = copy.deepcopy(self.left_img)
            r_img = copy.deepcopy(self.right_img)
            height = l_img.shape[0]
            self.img_lock.release()

            l_pose_t = infer_fast(self.net, l_img, height, self.stride, self.upsample_ratio, self.cpu)
            r_pose_t = infer_fast(self.net, r_img, height, self.stride, self.upsample_ratio, self.cpu)

            self.pose_lock.acquire()
            self.l_pose = copy.deepcopy(l_pose_t)
            self.r_pose = copy.deepcopy(r_pose_t)
            self.l_pose_img = copy.deepcopy(l_img)
            self.r_pose_img = copy.deepcopy(r_img)

            self.l_newpose = True
            self.r_newpose = True
            self.pose_lock.release()

        print("infering thread stopped")

    def set_images(self, l_img, r_img):
        self.img_lock.acquire()
        self.left_img = l_img
        self.right_img = r_img
        self.height = self.left_img.shape[0]
        self.img_lock.release()

    def get_left_pose(self):
        self.pose_lock.acquire()
        pose = copy.deepcopy(self.l_pose)
        img = copy.deepcopy(self.l_pose_img)
        posestatus = self.l_newpose
        self.pose_lock.release()

        self.l_newpose = False
        return posestatus, pose, img

    def get_right_pose(self):
        self.pose_lock.acquire()
        pose = copy.deepcopy(self.r_pose)
        img = copy.deepcopy(self.r_pose_img)
        posestatus = self.r_newpose
        self.pose_lock.release()

        self.r_newpose = False
        return posestatus, pose, img
        
