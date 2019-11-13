"""
Created on 19.11.12 21:13
@File:detect.py
@author: coderwangson
"""
"#codeing=utf-8"
import cv2
import time
from scipy.spatial import distance
delay_time = 6
blink_times = 4
type_txt = ["blink eye","open mouth","node head","shake head"]
type_action = ["eye","mouth","node","shake"]
def detect(face,cap,type):
    """face 是带状态的面部 cap是视频流 type是需要识别的类型4种"""
    blink = 0
    s = time.time()

    while True:
        ret, img = cap.read()  # 读取视频流的一帧
        img = cv2.flip(img, 1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        circle_center = (w // 2, h // 2)
        circle_r = min(h, w) // 3
        cv2.circle(img, circle_center, circle_r, color=(0, 0, 255))
        cv2.putText(img, type_txt[type], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", img)
        face.get_action(image)
        if face.action[type_action[type]] == 1:
            blink += 1
        e = time.time()
        if blink == blink_times:  # 成功验证
            return True
        if e - s > delay_time:  # 超过十秒
            return False
        cv2.waitKey(24)


def adjust_pose(face,cap):
    def point_distance(p1,p2,dis):
        """检测p1 和p2之间距离是否小于dis"""
        return distance.euclidean(p1,p2) < dis

    while True:
        ret, img = cap.read()  # 读取视频流的一帧
        img = cv2.flip(img,1)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face.get_action(image)
        h,w ,c= image.shape
        cv2.putText(img, "adjust your pose in the circle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        circle_center = (w//2,h//2)
        circle_r = min(h,w)//3
        cv2.circle(img,circle_center,circle_r,color=(0,0,255))
        cv2.imshow("Frame", img)
        cv2.waitKey(24)
        if face.state["node"] == 0 and face.state["shake"] ==0 and point_distance(face.face_center,circle_center,circle_r//2):
            return True

