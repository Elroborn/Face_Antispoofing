# Face_Antispoofing


---

本文是项目https://github.com/coderwangson/Face_Antispoofing 的说明。 

##  依赖环境  

opencv  
tensorflow  
torch  
dlib  

## 项目简介

本项目提供了带交互的活体检测，通过输出指定指令，让用户配合，如果没有按照指令做出相应动作，则判定为负样本。  

提供的指令包括`眨眼(blink eye) 张嘴(open mouth) 点头(node head) 摇头(shake head)`，每次随机从四个指令里面选出来3个进行交互。  

对于眨眼检测基于 https://blog.csdn.net/hongbin_xu/article/details/79033116 提供的方案。  

对于张嘴检测则基于 https://github.com/mauckc/mouth-open 。  

因为这些检测都是基于面部的landmark，而mtcnn等提供的landmark容易跟丢，所以本项目使用的是 https://github.com/610265158/Peppa_Pig_Face_Engine 提供的landmark检测方案，模型在这个项目里面也有给出，只需要下载Peppa提供的模型并放在model里面即可。  

## 项目运行  

* 下载 https://github.com/610265158/Peppa_Pig_Face_Engine#train 提供的模型  
* python main.py即可  


默认调用的是电脑摄像头，需要先把面部放到圆圈里面对齐，然后按照左上角的指令进行指定动作即可。  

可能由于摄像头型号不同，导致对应眨眼或者摇头的阈值不同，则可以在`face.py`上面进行配置。




