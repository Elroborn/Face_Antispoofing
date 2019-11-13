import cv2
from face import Face
import detect
import numpy as np
detect_type = [0,1,2,3] #分别代表["eye_detect","mouth_detect","node_detect","shake_detect"]
step_pass = [False,False,False] # 三步验证
face = Face()
cap = cv2.VideoCapture(0)
np.random.shuffle(detect_type)



detect.adjust_pose(face,cap)

for i,type in enumerate(detect_type[0:3]):
    step_pass[i] = detect.detect(face,cap,type) #getattr(detect, fun)(face, cap)


is_pass = (step_pass[0] and step_pass[1] and step_pass[2])
while True:
    ret, img = cap.read()  # 读取视频流的一帧
    img = cv2.flip(img, 1)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if is_pass:
        cv2.putText(img, "Pass ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Reject ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", img)
    if cv2.waitKey(24) & 0xFF == ord('q'):
        break