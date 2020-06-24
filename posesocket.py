import numpy as np
import socket


class PoseTransmitter:
    def __init__(self, host="127.0.0.1", port=1234):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen()
        
        self.connection = None
        self.address = None
        

    def await_connection(self):
        print("awaiting connection...")
        self.connection, self.address = self.socket.accept()
        print("connected")

    def pose_to_string(self, pose, pose_id):
        pose = pose[0]
        poses = "pose_id:" + str(pose_id) + ";"
        for i in range(len(pose)):
            try:
                cjoint = str(i) + ":" + "{:.5f}".format(pose[i][0]) + ":" + "{:.5f}".format(pose[i][1]) + ":" + "{:.5f}".format(pose[i][2]) + ";"
            except:
                print("couldn't parse pose", pose[i])
                cjoint = ""
                # quit()
            poses += cjoint

        poses += "pose_end;"

        return poses

    def pose_to_bytes(self, pose):
        tpose = "begin_stream;"

        for pose_id in range(len(pose)):
            strpose = self.pose_to_string(pose[pose_id], pose_id)
            tpose += strpose

        tpose += "end_stream;"
        # print(tpose)
        bytepose = tpose.encode('utf-8')

        return bytepose

    def transmit_pose(self, pose):
        bytepose = self.pose_to_bytes(pose)

        if self.connection is None or self.address is None:
            self.await_connection()
        
        try:
            self.connection.sendall(bytepose)
        except:
            self.connection = None
            self.address = None
