import dlib
import numpy as np
import cv2
import sys


K = np.array([[3.082395817372530e+03, 0.0, 1.033856512831924e+03],
              [0.0, 3.078037901493608e+03, 4.855400411851134e+02],
              [0.0, 0.0, 1.0]], np.float64)

D = np.array([-0.408381989766528, -0.269697366983746, 0.0, 0.0], np.float64)

R_screenInCam = np.array([[-0.99963107, 0.02541517, -0.00958103],
                          [-0.02676235, -0.98186443, 0.18768595],
                          [0.00463721, -0.18787311, -0.98218236]], np.float64)
t_screenInCam = np.array([[0.13111697], [0.2226758], [0.13710232]], np.float64)

R_camInScreen = np.array([[-0.99963107, -0.02676235, 0.00463721],
                          [0.02541517, -0.98186443, -0.18787311],
                          [-0.00958103, 0.18768595, -0.98218236]])
t_camInScreen = np.array([[0.13639216], [0.24106293], [0.0941226]], np.float64)


class Camera:

    def __init__(self, intrinsicMat=K, distortionCof=D, imgWidth=1920, imgHeight=1080):
        self.w = imgWidth
        self.h = imgHeight
        self.K = intrinsicMat
        self.D = distortionCof
        self.R_camInScreen = R_camInScreen
        self.t_camInScreen = t_camInScreen
        self.predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()

    def SetCap(self,camId=0,w=600,h=400):
        self.cap=cv2.VideoCapture()
        self.cap.open(camId)
        # fourcc = cv2.CAP_PROP_FOURCC('M','J','P','G')
        self.cap.set(cv2.CAP_PROP_FOURCC,1196444237);
        #cv2.CAP_PROP_FOURCC('M','J','P','G'),设置视频读取格式为MJPG,cv2的python接口相比于c++此处有问题
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)


    def ImgCapture(self):
        value,frame=self.cap.read()
        return frame

    def LandmarkDetection(self,frame):
        """
        :param frame:   image frame
        :return:
            shape: landmark points using dlib calculate
            frame: landmark points on image
        """
        dets=self.detector(frame,1)
        if len(dets)!=0:
            shape=self.predictor(frame,dets[0])
            for p in shape.parts():
                cv2.circle(frame, (p.x, p.y), 3, (0, 255, 0), -1)
                print(type(shape.parts()))
            return shape,frame
        else:
            return 0,frame


class Screen:
    def __init__(self, rotationMatrixInCamera=R_screenInCam, transVectorInCamra=t_screenInCam):
        self.R_inCam = rotationMatrixInCamera
        self.t_inCam = transVectorInCamra

    def ScreenPoseEstimation(self, rvec, tvec, camera):
        """
        implement of paper "Camera Pose Estimation Using Images of Planar Mirror Reflections"
        caculate the screen pose under camera coordinate or camera pose under screen coordinate.

        :param rvec:    screen rotation vector under camera coordinate  (nx3 numpy ndarray)
        :param tvec:    screen translate vector under camera coordinate (nx3 numpy ndarray)
        :param camera:  Camera class
        :return:    none

        """

        """
        virtual screen or virtual camera pose parameters:        
        
        sA:     screen rotation Axis under camera coordinate
        sT:     screen translate vector under camera coordinate
        sR:     screen rotation  matrix *                       *
        sRI:    inverse of sR
        sTI:    inverse of sT
        cA:     camera rotation axis under screen coordinate
        cAa:    camera rotation angle 
        cR:     camera rotation Matrix
        cT:     camera rotation translate vector
        
        RESULTS:
        self.R_inCam:
        self.t_inCAM:
        camera.R_camInScreen = R
        camera.t_camInScreen = t
        
        """


        sA = rvec
        sT = tvec
        for i in range(sA.shape[0]):
            if i == 0:
                sR, _ = cv2.Rodrigues(sA[0])
                sRI = np.linalg.inv(sR)
                sTI = np.matmul(-sRI, np.transpose(sT[0])).reshape((1, 3))
                cR = sR.copy()
                cT = sT[0].reshape((1, 3))

                tmpRI0I = sR
                tmpTI0 = np.transpose(sTI)
                cA, _ = cv2.Rodrigues(cR)
                cAa = (np.linalg.norm(cA)).reshape((1,1))
                # cAa = cAa.reshape((1, 1))
                cA = np.transpose(cA) / cAa
            else:
                tmpsR, _ = cv2.Rodrigues(sA[i])
                tmpsRI = np.linalg.inv(tmpsR)
                tmpsTI = (-np.matmul(tmpsRI, np.transpose(sT[i]).reshape((3, 1)))).reshape((1, 3))
                tmpcR = np.matmul(tmpsRI, tmpRI0I)
                tmpcT = (tmpsTI - np.transpose(np.matmul(tmpcR, tmpTI0))).reshape((1, 3))
                tmpcA, _ = cv2.Rodrigues(tmpcR)
                tmpcAa = np.linalg.norm(tmpcA)
                tmpcAa = tmpcAa.reshape((1, 1))
                tmpcA = tmpcA.reshape((1, 3)) / tmpcAa

                sR = np.concatenate((sR, tmpsR), 0)
                sRI = np.concatenate((sRI, tmpsRI), 0)
                sTI = np.concatenate((sTI, tmpsTI), 0)
                cR = np.concatenate((cR, tmpcR), 0)
                cT = np.concatenate((cT, tmpcT), 0)
                cA = np.concatenate((cA, tmpcA), 0)
                cAa = np.concatenate((cAa, tmpcAa), 0)

        for i in range(1, sA.shape[0]):
            if i == 1:
                A1 = cross(cT[i]) + np.tan(cAa[i] / 2.0) * np.matmul(cA[i].reshape((3, 1)),
                                                                     np.transpose(cT[i]).reshape((1, 3)))
                A2 = (-2 * np.tan(cAa[i] / 2.0) * cA[i]).reshape((3, 1))
                A = np.concatenate((A1, A2), 1)
            else:
                A1 = cross(cT[i]) + np.tan(cAa[i] / 2.0) * np.matmul(cA[i].reshape((3, 1)),
                                                                     np.transpose(cT[i]).reshape((1, 3)))
                A2 = (-2 * np.tan(cAa[i] / 2.0) * cA[i]).reshape((3, 1))
                B = np.concatenate((A1, A2), 1)
                A = np.concatenate((A, B), 0)

        U, Sigma, V = np.linalg.svd(A)
        n_norm = np.linalg.norm(V[3, 0:3])
        n = (V[3, 0:3] / n_norm).reshape(3, 1)
        d = V[3, 3] / n_norm
        I = np.eye(3)
        tmpR = I - 2 * np.matmul(n, np.transpose(n))
        tmpt = 2 * d * n

        # 屏幕坐标系下相机的位姿
        R = np.matmul(tmpR, np.linalg.inv(cR[0:3, ]))
        t = tmpt - np.matmul(R, cT[0].reshape((3, 1)))
        camera.R_camInScreen = R
        camera.t_camInScreen = t
        # 相机坐标系下屏幕的位姿
        RR = np.linalg.inv(R)
        tt = -np.matmul(np.linalg.inv(R), t)
        self.R_inCam = RR
        self.t_inCam = tt

        return 0


def cross(a):
    a = a.reshape((1, 3))
    return np.array([[0, -a[0][2], a[0][1]], [a[0, 2], 0, -a[0, 0]], [-a[0, 1], a[0, 0], 0]])


class Head:
    def __init__(self, landmark3d, landmark2d, pic, eyeCornerLen):
        self.ratio = eyeCornerLen / np.sqrt((landmark2d[36, 0] - landmark2d[45, 0]) ** 2 + (landmark2d[36, 1] - landmark2d[45, 1]) ** 2)
        self.point3d = self.ratio*(landmark3d-(landmark3d[36]+landmark3d[45])/2)
        self.point2d = landmark2d
        self.pic = pic
        self.eyeCornerLen=eyeCornerLen




    def HeadPoseEstimation(self,camera,screen):
        point2d=np.ascontiguousarray(self.point2d.reshape((self.point2d.shape[0],2,1)))
        point3d=np.ascontiguousarray(self.point3d.reshape((self.point3d.shape[0],3,1)))
        a,b,c=cv2.solvePnP(point3d,point2d,cameraMatrix=camera.K,distCoeffs=camera.D)
        headT = c.reshape((3, 1))
        headR, _ = cv2.Rodrigues(b)
        transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        camera.R_inHead=headR
        camera.t_inHead=headT

        self.R_inCam=np.linalg.inv(headR)
        self.t_inCam=-np.matmul(self.R_inCam,headT)
        Rheads = np.matmul(np.matmul(screen.R_inCam, transform), headR)
        Theads = np.matmul(np.matmul(screen.R_inCam, transform), headT) +screen.t_inCam

        self.R_inScreen = np.linalg.inv(Rheads)
        self.t_inScreen = -np.matmul(self.R_inScreen, Theads)

        landmark3dInCam=self.point3d
        for i in range(self.point3d.shape[0]):
            landmark3dInCam[i]=(np.matmul(self.R_inScreen,landmark3dInCam[i].reshape((3,1)))+self.t_inScreen).reshape((1,3))
        return landmark3dInCam

    def ImgNormalization(self):
        return 0





