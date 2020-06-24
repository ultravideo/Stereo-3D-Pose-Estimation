import cv2
import numpy as np
import time


def main():

    captime = 30
    initdelay = 4
    begintime = time.time()

    Lcam = cv2.VideoCapture(6)
    Rcam = cv2.VideoCapture(4)

    print("waiting...")
    while time.time() - begintime < initdelay:
        continue

    Lframes = []
    Rframes = []

    height, width, channels = 0, 0, 0

    fps = 30
    frametime = 1 / fps

    print("begin capture")

    begintime = time.time()
    while True:

        Limg = Lcam.read()[1]
        Rimg = Rcam.read()[1]

        Lframes.append(Limg)
        Rframes.append(Rimg)

        cv2.imshow("img", np.hstack((Limg, Rimg)))
        cv2.waitKey(1)

        if time.time() - begintime > captime:
            print("captured")
            height, width, channels = Limg.shape
            break

        if time.time() - begintime < frametime:
            continue
        

    print("writing video")

    Lvwriter = cv2.VideoWriter("L_video_multi_15cm.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    Rvwriter = cv2.VideoWriter("R_video_multi_15cm.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for i in range(len(Lframes)):
        Lvwriter.write(Lframes[i])
        Rvwriter.write(Rframes[i])

    Lvwriter.release()
    Rvwriter.release()

    print("videos written")


main()
