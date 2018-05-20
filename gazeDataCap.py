import numpy as np
import gazeClass as gc
import cv2
import face_alignment
from skimage import io
import dlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math


def get_3dLandmarks_from_face_alignment(preds):
    """
    :param preds:output of face_alignment.get_landmarks,which are 3d face landmarks but the coordiante center is not head
    :return:  3d face landmarks which the coordiante center is head center
    """
    preds2 = preds.copy()
    preds2 = preds - (preds[45] + preds[36]) / 2
    theta = -math.atan(preds2[36, 1] / preds2[36, 0])
    Rvec = np.array([0, 0, 1], np.float64) * theta
    R, _ = cv2.Rodrigues(Rvec)
    preds3 = preds2
    for i in range(preds2.shape[0]):
        preds3[i] = np.matmul(R, preds2[i].reshape((3, 1))).reshape((1, 3))
    return preds3


def plt_landmark2d(pic, landmark2d):
    fig = plt.figure()
    # ax=Axes3D(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(landmark2d[0:17, 0], landmark2d[0:17, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2d[17:22, 0], landmark2d[17:22, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2d[22:27, 0], landmark2d[22:27, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2d[27:31, 0], landmark2d[27:31, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2d[31:36, 0], landmark2d[31:36, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2d[36:42, 0], landmark2d[36:42, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2d[42:48, 0], landmark2d[42:48, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2d[48:60, 0], landmark2d[48:60, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2d[60:68, 0], landmark2d[60:68, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    # for i in range(landmark2d.shape[0]):
    #     ax.text(landmark2d[i][0],landmark2d[i][1],s=str(i))
    plt.title('2d facial landmarks in picture')
    ax.imshow(pic)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plt_landmark3d(landmark3d):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.scatter(landmark3d[:, 0] * 1.0, landmark3d[:, 1], landmark3d[:, 2], c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(landmark3d[:17, 0] * 1.0, landmark3d[:17, 1], landmark3d[:17, 2], color='blue')
    ax.plot3D(landmark3d[17:22, 0] * 1.0, landmark3d[17:22, 1], landmark3d[17:22, 2], color='blue')
    ax.plot3D(landmark3d[22:27, 0] * 1.0, landmark3d[22:27, 1], landmark3d[22:27, 2], color='blue')
    ax.plot3D(landmark3d[27:31, 0] * 1.0, landmark3d[27:31, 1], landmark3d[27:31, 2], color='blue')
    ax.plot3D(landmark3d[31:36, 0] * 1.0, landmark3d[31:36, 1], landmark3d[31:36, 2], color='blue')
    ax.plot3D(landmark3d[36:42, 0] * 1.0, landmark3d[36:42, 1], landmark3d[36:42, 2], color='blue')
    ax.plot3D(landmark3d[42:48, 0] * 1.0, landmark3d[42:48, 1], landmark3d[42:48, 2], color='blue')
    ax.plot3D(landmark3d[48:, 0] * 1.0, landmark3d[48:, 1], landmark3d[48:, 2], color='blue')
    ax.scatter(landmark3d[36, 0], landmark3d[36, 1], landmark3d[36, 2], c='r', s=40)
    ax.scatter(0, 0, 0, c='r', s=20)

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_zlim(-400, 400)
    plt.title('3d facial landmark model')

    ax.set_aspect(1)


def plt_face_screen_cam(landmark3dInCam, screen, camera):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    X = np.array([0, 265, 265, 0])
    Y = np.array([0, 0, 185, 185])
    X, Y = np.meshgrid(X, Y)
    Z = X * 0
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.gray, alpha=0.5)

    ax.scatter(camera.t_camInScreen[0] * 1000, camera.t_camInScreen[1] * 1000, camera.t_camInScreen[2] * 1000, c="r",
               s=10)
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

    plt.title('face and camera location under screen coordinate')
    ax.set_xlabel(xlabel="X (mm)")
    ax.set_ylabel(ylabel="Y (mm)")
    ax.set_zlabel(zlabel="Z (mm)")


def calculateNormalizationMatrix(head, camera):
    Crinv = np.linalg.inv(camera.K)
    R = head.R_inCam
    Cn = camera.K
    M = np.matmul(np.matmul(Cn, R), Crinv)
    return M


def imgNormalization(pic, landmark2d, M):
    picNorm = cv2.warpPerspective(pic, M, (1920, 1080))

    landmarkNorm = landmark2d.copy()
    for i in range(landmark2d.shape[0]):
        zPoint = (M[2][0] * landmark2d[i][0] + M[2][1] * landmark2d[i][1] + M[2][2])
        xPoint = (M[0][0] * landmark2d[i][0] + M[0][1] * landmark2d[i][1] + M[0][2]) / zPoint
        yPoint = (M[1][0] * landmark2d[i][0] + M[1][1] * landmark2d[i][1] + M[1][2]) / zPoint
        landmarkNorm[i][0] = xPoint
        landmarkNorm[i][1] = yPoint
    return picNorm, landmarkNorm


def plt_normalized_landmark2d(picNorm, landmark2dNorm):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(landmark2dNorm[0:17, 0], landmark2dNorm[0:17, 1], marker='o', markersize=3, linestyle='-', color='w', lw=2)
    ax.plot(landmark2dNorm[17:22, 0], landmark2dNorm[17:22, 1], marker='o', markersize=3, linestyle='-', color='w',
            lw=2)
    ax.plot(landmark2dNorm[22:27, 0], landmark2dNorm[22:27, 1], marker='o', markersize=3, linestyle='-', color='w',
            lw=2)
    ax.plot(landmark2dNorm[27:31, 0], landmark2dNorm[27:31, 1], marker='o', markersize=3, linestyle='-', color='w',
            lw=2)
    ax.plot(landmark2dNorm[31:36, 0], landmark2dNorm[31:36, 1], marker='o', markersize=3, linestyle='-', color='w',
            lw=2)
    ax.plot(landmark2dNorm[36:42, 0], landmark2dNorm[36:42, 1], marker='o', markersize=3, linestyle='-', color='w',
            lw=2)
    ax.plot(landmark2dNorm[42:48, 0], landmark2dNorm[42:48, 1], marker='o', markersize=3, linestyle='-', color='w',
            lw=2)
    ax.plot(landmark2dNorm[48:60, 0], landmark2dNorm[48:60, 1], marker='o', markersize=3, linestyle='-', color='w',
            lw=2)
    ax.plot(landmark2dNorm[60:68, 0], landmark2dNorm[60:68, 1], marker='o', markersize=3, linestyle='-', color='w',
            lw=2)
    ax.imshow(picNorm)


def get_eye_region(picNorm, landmark2dNorm):
    laa = int(landmark2dNorm[39][0] - landmark2dNorm[36][0])
    lbb = int(landmark2dNorm[36][0] - laa / 2)
    lcc = int(landmark2dNorm[39][0] + laa / 2)
    ldd = int((landmark2dNorm[36][1] + landmark2dNorm[39][1]) / 2 - laa / 2)
    lee = int((landmark2dNorm[36][1] + landmark2dNorm[39][1]) / 2 + laa / 2)
    leye = picNorm[ldd:lee, lbb:lcc]

    aa = int(landmark2dNorm[45][0] - landmark2dNorm[42][0])
    bb = int(landmark2dNorm[42][0] - aa / 2)
    cc = int(landmark2dNorm[45][0] + aa / 2)
    dd = int((landmark2dNorm[45][1] + landmark2dNorm[42][1]) / 2 - aa / 2)
    ee = int((landmark2dNorm[45][1] + landmark2dNorm[42][1]) / 2 + aa / 2)
    reye = picNorm[dd:ee, bb:cc]
    return leye, reye


def plt_eyes(leye, reye):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(leye)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(reye)
    # Save result
    # out_name1 = input_name + '_out1.png'
    # out_name2 = input_name + '_eye.png'
    # fig.savefig('./result/' + out_name1, format='png')
    # fig2.savefig('./result/' + out_name2, format='png')


# Read camera  parameters(intrinsic and extrinsic parameters)
dtype = np.float64
fs = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
para = np.array(fs.getNode("extrinsic_parameters").mat(), np.float64)

# Compute screen pose
camera = gc.Camera()
screen = gc.Screen()
camera.R_camInScreen, camera.t_camInScreen = screen.ScreenPoseEstimation(para[:, 0:3], para[:, 3:6], camera)

# Get 3d facial landmark model
input_name = 'mwm'
pic = io.imread('./pic/' + input_name + '.jpg')
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)
preds = fa.get_landmarks(pic)[-1]
landmark3d = get_3dLandmarks_from_face_alignment(preds)

# Get 2d facial landmark from current picture
landmark2d = preds[:, 0:2]

head = gc.Head(landmark3d, landmark2d, pic, 0.1176)
landmark3dInCam = head.HeadPoseEstimation(camera, screen)

# Calculate Head position under camera(or screen) coordinate with PnP algorithm
camera.SetCap()

# Image normalization and get eye region
M = calculateNormalizationMatrix(head, camera)
picNorm, landmark2dNorm = imgNormalization(pic, landmark2d, M)
leye, reye = get_eye_region(picNorm, landmark2dNorm)

# Result show
plt_landmark2d(pic, landmark2d)
plt_landmark3d(landmark3d)
plt_face_screen_cam(landmark3dInCam, screen, camera)
plt_normalized_landmark2d(picNorm, landmark2dNorm)
plt_eyes(leye, reye)
plt.show()

print(pic.shape)
