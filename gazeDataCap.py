import numpy as np
import gazeClass as gc
import cv2
import face_alignment
from skimage import io
import dlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

dtype=np.float64
fs = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
para = np.array(fs.getNode("extrinsic_parameters").mat(), np.float64)

camera = gc.Camera()
screen = gc.Screen()
screen.ScreenPoseEstimation(para[:, 0:3], para[:, 3:6], camera)

pic = io.imread('./pic/mwm.jpg')
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)
preds = fa.get_landmarks(pic)[-1]

preds2 = preds.copy()
preds2 = preds - (preds[45] + preds[36]) / 2
theta = -math.atan(preds2[36, 1] / preds2[36, 0])
Rvec = np.array([0, 0, 1], np.float64) * theta
R, _ = cv2.Rodrigues(Rvec)


preds3 = preds2.copy()
for i in range(preds2.shape[0]):
    preds3[i]=np.matmul(R, preds2[i].reshape((3, 1))).reshape((1, 3))

landmark3d = preds3.copy()
# 3
# [[ 0.95185135 -0.01023712 -0.30638899]
#  [-0.02068299  0.99502047 -0.09750113]
#  [ 0.30586145  0.09914362  0.94689985]]
# 2
# [[ 0.95185135 -0.01023712 -0.30638899]
#  [-0.02068299  0.99502047 -0.09750113]
#  [ 0.30586145  0.09914362  0.94689985]]
# print(preds3-preds2)
landmark2d = preds[:, 0:2]

head = gc.Head(landmark3d, landmark2d, pic, 0.1176)
landmark3dInCam = head.HeadPoseEstimation(camera, screen)

camera.SetCap()
# predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
# # detector = dlib.get_frontal_face_detector()
#
# while True:
#     frame = camera.ImgCapture()
#     shape,frame=camera.LandmarkDetection(frame)
#     cv2.imshow('video', frame)
#     cv2.waitKey(30)
# # print(1)
# # print('M'+'J'+'P'+'G')

#########################################################################
fig = plt.figure()
# ax=Axes3D(fig)
ax = fig.add_subplot(2, 2, 1, projection='3d')
X = np.array([0, 265, 265, 0])
Y = np.array([0, 0, 185, 185])
X, Y = np.meshgrid(X, Y)
Z = X * 0
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.gray, alpha=0.5)

ax.scatter(camera.t_camInScreen[0] * 1000, camera.t_camInScreen[1] * 1000, camera.t_camInScreen[2] * 1000, c="r", s=10)
landmark3dInCam = landmark3dInCam * 1000
ax.scatter(landmark3dInCam[:, 0], landmark3dInCam[:, 1], landmark3dInCam[:, 2], s=1)
ax.plot3D(landmark3dInCam[:17, 0] * 1.0, landmark3dInCam[:17, 1], landmark3dInCam[:17, 2], color='blue')
ax.plot3D(landmark3dInCam[17:22, 0] * 1.0, landmark3dInCam[17:22, 1], landmark3dInCam[17:22, 2], color='blue')
ax.plot3D(landmark3dInCam[22:27, 0] * 1.0, landmark3dInCam[22:27, 1], landmark3dInCam[22:27, 2], color='blue')
ax.plot3D(landmark3dInCam[27:31, 0] * 1.0, landmark3dInCam[27:31, 1], landmark3dInCam[27:31, 2], color='blue')
ax.plot3D(landmark3dInCam[31:36, 0] * 1.0, landmark3dInCam[31:36, 1], landmark3dInCam[31:36, 2], color='blue')
ax.plot3D(landmark3dInCam[36:42, 0] * 1.0, landmark3dInCam[36:42, 1], landmark3dInCam[36:42, 2], color='blue')
ax.plot3D(landmark3dInCam[42:48, 0] * 1.0, landmark3dInCam[42:48, 1], landmark3dInCam[42:48, 2], color='blue')
ax.plot3D(landmark3dInCam[48:, 0] * 1.0, landmark3dInCam[48:, 1], landmark3dInCam[48:, 2], color='blue')
ax.set_aspect(1)
ax.set_zlim(-1500, 500)
ax.set_xlim(-100, 500)
ax.set_ylim(-100, 500)
ax.view_init(elev=90, azim=-90)

ax.set_xlabel(xlabel="X (mm)")
ax.set_ylabel(ylabel="Y (mm)")
ax.set_zlabel(zlabel="Z (mm)")
#####################################################################
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(preds[0:17, 0], preds[0:17, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax2.plot(preds[17:22, 0], preds[17:22, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax2.plot(preds[22:27, 0], preds[22:27, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax2.plot(preds[27:31, 0], preds[27:31, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax2.plot(preds[31:36, 0], preds[31:36, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax2.plot(preds[36:42, 0], preds[36:42, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax2.plot(preds[42:48, 0], preds[42:48, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax2.plot(preds[48:60, 0], preds[48:60, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax2.plot(preds[60:68, 0], preds[60:68, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
# for i in range(preds.shape[0]):
#     ax2.text(preds[i][0],preds[i][1],s=str(i))
ax2.imshow(pic)
ax2.set_xlabel("x")
ax2.set_ylabel("y")

#####################################################################
preds4=preds3.copy()
for i in range(preds4.shape[0]):
    preds4[i] = np.matmul(head.R_inCam, preds4[i].reshape((3, 1))).reshape((1, 3))

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
surf = ax3.scatter(preds4[:, 0] * 1.0, preds4[:, 1], preds4[:, 2], c="cyan", alpha=1.0, edgecolor='b')
ax3.plot3D(preds4[:17, 0] * 1.0, preds4[:17, 1], preds4[:17, 2], color='blue')
ax3.plot3D(preds4[17:22, 0] * 1.0, preds4[17:22, 1], preds4[17:22, 2], color='blue')
ax3.plot3D(preds4[22:27, 0] * 1.0, preds4[22:27, 1], preds4[22:27, 2], color='blue')
ax3.plot3D(preds4[27:31, 0] * 1.0, preds4[27:31, 1], preds4[27:31, 2], color='blue')
ax3.plot3D(preds4[31:36, 0] * 1.0, preds4[31:36, 1], preds4[31:36, 2], color='blue')
ax3.plot3D(preds4[36:42, 0] * 1.0, preds4[36:42, 1], preds4[36:42, 2], color='blue')
ax3.plot3D(preds4[42:48, 0] * 1.0, preds4[42:48, 1], preds4[42:48, 2], color='blue')
ax3.plot3D(preds4[48:, 0] * 1.0, preds4[48:, 1], preds4[48:, 2], color='blue')
ax3.scatter(preds4[36, 0], preds4[36, 1], preds4[36, 2], c='r', s=40)
ax3.scatter(0, 0, 0, c='r', s=20)


ax3.view_init(elev=90., azim=90.)
ax3.set_xlim(ax3.get_xlim()[::-1])
ax3.set_xlim(-400, 400)
ax3.set_ylim(-400, 400)
ax3.set_zlim(-400, 400)


ax3.set_aspect(1)
#####################################################################
ax4=fig.add_subplot(2,2,4)
trans=(landmark2d[36]+landmark2d[45])/2

theta = -math.atan(preds2[36, 1] / preds2[36, 0])

Crinv = np.linalg.inv(camera.K)
R = head.R_inCam
print(R)
Cn = camera.K
M = np.matmul(np.matmul(Cn, R), Crinv)
print(M)
pic2 = cv2.warpPerspective(pic, M, (1920,1080))

landmark2dNew=landmark2d.copy()
for i in range(landmark2d.shape[0]):
        zPoint=(M[2][0]*landmark2d[i][0]+M[2][1]*landmark2d[i][1]+M[2][2])
        xPoint=(M[0][0]*landmark2d[i][0]+M[0][1]*landmark2d[i][1]+M[0][2])/zPoint
        yPoint=(M[1][0]*landmark2d[i][0]+M[1][1]*landmark2d[i][1]+M[1][2])/zPoint
        landmark2dNew[i][0]=xPoint
        landmark2dNew[i][1]=yPoint


ax4.plot(landmark2dNew[0:17, 0], landmark2dNew[0:17, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax4.plot(landmark2dNew[17:22, 0], landmark2dNew[17:22, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax4.plot(landmark2dNew[22:27, 0], landmark2dNew[22:27, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax4.plot(landmark2dNew[27:31, 0], landmark2dNew[27:31, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax4.plot(landmark2dNew[31:36, 0], landmark2dNew[31:36, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax4.plot(landmark2dNew[36:42, 0], landmark2dNew[36:42, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax4.plot(landmark2dNew[42:48, 0], landmark2dNew[42:48, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax4.plot(landmark2dNew[48:60, 0], landmark2dNew[48:60, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
ax4.plot(landmark2dNew[60:68, 0], landmark2dNew[60:68, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)

ax4.imshow(pic2)

#####################################################################
fig2=plt.figure()
laa=int(landmark2dNew[39][0]-landmark2dNew[36][0])
lbb=int(landmark2dNew[36][0]-laa/2)
lcc=int(landmark2dNew[39][0]+laa/2)
ldd=int((landmark2dNew[36][1]+landmark2dNew[39][1])/2-laa/2)
lee=int((landmark2dNew[36][1]+landmark2dNew[39][1])/2+laa/2)
leye=pic2[ldd:lee,lbb:lcc]
fig2_ax2=fig2.add_subplot(2,2,1)
fig2_ax2.imshow(leye)
#####################################################################
aa=int(landmark2dNew[45][0]-landmark2dNew[42][0])
bb=int(landmark2dNew[42][0]-aa/2)
cc=int(landmark2dNew[45][0]+aa/2)
dd=int((landmark2dNew[45][1]+landmark2dNew[42][1])/2-aa/2)
ee=int((landmark2dNew[45][1]+landmark2dNew[42][1])/2+aa/2)
reye=pic2[dd:ee,bb:cc]
fig2_ax1=fig2.add_subplot(2,2,2)
fig2_ax1.imshow(reye)
#####################################################################
plt.show()
print(pic.shape)
