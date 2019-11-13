"""
Created on 19.11.12 14:18
@File:Face.py
@author: coderwangson
"""
"#codeing=utf-8"
from scipy.spatial import distance
import copy
from lib.core.api.facer import FaceAna
from lib.core.headpose.pose import get_head_pose

# eye config
# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1
EYE_AR_THRESH = 0.15 # EAR阈值

# mouth config
MOUTH_AR_THRESH = 0.79
(mStart, mEnd) = (49, 68)

# head config
NODE_THRESH = 2
SHAKE_THRESH = 25

facer = FaceAna()
class Face(object):
    def __init__(self):
        # eye mouth： 0 是闭着， 1 是张开
        # head中node -1 代表点头 1 代表抬头 0 代表正面
        # shake -1 代表左摇头 1代表右边摇头
        self.state = {"eye": 1, "mouth": 0,"node": 0,"shake": 0}

        self.last_state = {"eye": 1, "mouth": 0,"node": 0,"shake": 0}  # 表示上一帧的动作
        # eye 0 代表没有动作 1代表眨眼动作 mouth 0 代表无动作，1代表张闭嘴动作
        # head node 0代表无动作，1代表点头/抬头动作 shake类似
        self.action = {"eye": 0, "mouth": 0,"node": 0,"shake": 0}
        self.image = None



    def update_state(self):

        boxes, landmarks, states = facer.run(self.image)

        self.boxes = boxes[0]
        self.face_center = ((boxes[0][0] + boxes[0][2])//2 ,(boxes[0][1] + boxes[0][3])//2)

        # update eye
        ear = self.__get_eye(landmarks)
        if ear < EYE_AR_THRESH:  # 代表闭眼
            self.state["eye"] = 0
        else:
            self.state["eye"] = 1

        # update mouth
        mar = self.__get_mouth(landmarks)
        if mar < MOUTH_AR_THRESH:  # 代表闭嘴
            self.state["mouth"] = 0
        else:
            self.state["mouth"] = 1

        # update head
        node,shake = self.__get_head(landmarks)

        if abs(node) < NODE_THRESH:
            self.state["node"] = 0
        else:
            self.state["node"] = 1

        if abs(shake) < SHAKE_THRESH:
            self.state["shake"] = 0
        else:
            self.state["shake"] = 1



    def get_action(self,image):
        """image is rgb image"""

        if self.image is None: # 第一次使用
            self.image = image
            self.update_state()
            self.last_state = copy.deepcopy(self.state)
        else: # 非第一次使用，即已经有了last_state
            self.image = image
            self.last_state = copy.deepcopy(self.state)
            self.update_state()

        if self.state["eye"] != self.last_state["eye"]:
            self.action["eye"] = 1
        else:
            self.action["eye"] = 0

        if self.state["mouth"] != self.last_state["mouth"]:
            self.action["mouth"] = 1
        else:
            self.action["mouth"] = 0

        if self.state["node"] != self.last_state["node"]:
            self.action["node"] = 1
        else:
            self.action["node"] = 0



        if self.state["shake"] != self.last_state["shake"]:
            self.action["shake"] = 1
        else:
            self.action["shake"] = 0

    def __get_eye(self,landmarks):
        leftEye = landmarks[0][LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼对应的特征点
        rightEye = landmarks[0][RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼对应的特征点

        def eye_aspect_ratio(eye):
            # print(eye)
            A = distance.euclidean(eye[1], eye[5])
            B = distance.euclidean(eye[2], eye[4])
            C = distance.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C)
            return ear

        leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR
        rightEAR = eye_aspect_ratio(rightEye)  # 计算右眼EAR
        ear = (leftEAR + rightEAR) / 2.0  # 求左右眼EAR的均值

        return ear


    def __get_mouth(self,landmarks):
        mouth = landmarks[0][mStart:mEnd]

        def mouth_aspect_ratio(mouth):
            # compute the euclidean distances between the two sets of
            # vertical mouth landmarks (x, y)-coordinates
            A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
            B = distance.euclidean(mouth[4], mouth[8])  # 53, 57

            # compute the euclidean distance between the horizontal
            # mouth landmark (x, y)-coordinates
            C = distance.euclidean(mouth[0], mouth[6])  # 49, 55

            # compute the mouth aspect ratio
            mar = (A + B) / (2.0 * C)

            # return the mouth aspect ratio
            return mar

        mar = mouth_aspect_ratio(mouth)

        return mar



    def __get_head(self,landmarks):

        reprojectdst, euler_angle = get_head_pose(landmarks[0], self.image)

        node = euler_angle[0, 0]
        shake = euler_angle[1, 0]
        return node,shake
