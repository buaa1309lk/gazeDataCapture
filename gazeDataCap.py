import numpy as np
import gazeClass as gc
import cv2
import face_alignment
from skimage import io
import dlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from gazeApi import *

# Read camera  parameters(intrinsic and extrinsic parameters)
dtype = np.float64
fs = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
para = np.array(fs.getNode("extrinsic_parameters").mat(), np.float64)

# Compute screen pose
camera = gc.Camera()
screen = gc.Screen()
camera.R_camInScreen, camera.t_camInScreen = screen.ScreenPoseEstimation(para[:, 0:3], para[:, 3:6], camera)

# Get 3d facial landmark model
input_name = 'lk'
pic = io.imread('./pic/' + input_name + '.jpg')
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)
preds = fa.get_landmarks(pic)[-1]
landmark3d = get_3dLandmarks_from_face_alignment(preds)

# input_name2 = 'lk3'
# pic2 = io.imread('./pic/' + input_name2 + '.jpg')
# fa2 = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)
# preds2 = fa2.get_landmarks(pic2)[-1]
# landmark2d=preds2[:,0:2]
# Get 2d facial landmark from current picture

camera.SetCap()
landmark2dtmp=preds[:,0:2]
frame=camera.ImgCapture()
while True:
    frame=camera.ImgCapture()
    dets=camera.detector(frame,1)
    if len(dets)!=0:
        shape=camera.predictor(frame,dets[0])
        i=0
        for p in shape.parts():
            landmark2dtmp[i][0]=p.x
            landmark2dtmp[i][1]=p.y
            cv2.circle(frame,(p.x,p.y),3,(0,255,0),-1)
            i=i+1
        break
    # cv2.imshow('video',frame)
    # cv2.waitKey(30)
landmark2d=landmark2dtmp



# Calculate Head position under camera(or screen) coordinate with PnP algorithm
head = gc.Head(landmark3d, landmark2d, frame, 0.1176)
landmark3dInCam = head.HeadPoseEstimation(camera, screen)

# Image normalization and get eye region
M = calculateNormalizationMatrix(head, camera)
picNorm, landmark2dNorm = imgNormalization(frame, landmark2d, M)
leye, reye = get_eye_region(picNorm, landmark2dNorm)

# Result show
plt_landmark2d(frame, landmark2d)
plt_landmark3d(landmark3d)
plt_face_screen_cam(landmark3dInCam, screen, camera)
plt_normalized_landmark2d(picNorm, landmark2dNorm)
# plt.imshow(leye)
print('eye shape is :')
print(leye.shape)
print(reye.shape)
plt_eyes(leye, reye)
plt.show()

print(pic.shape)
