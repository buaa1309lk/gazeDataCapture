from gazeApi import *
import datetime

# Read camera  parameters(intrinsic and extrinsic parameters)
dtype = np.float64
fs = cv2.FileStorage("camera.yaml", cv2.FILE_STORAGE_READ)
para = np.array(fs.getNode("extrinsic_parameters").mat(), np.float64)

# Compute screen pose
camera = gc.Camera()
screen = gc.Screen()
camera.R_camInScreen, camera.t_camInScreen = \
    screen.ScreenPoseEstimation(para[:, 0:3], para[:, 3:6], camera)

# Get 3d facial landmark model
input_name = 'lk'
pic = io.imread('./pic/' + input_name + '.jpg')
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)
preds = fa.get_landmarks(pic)[-1]
landmark3d = get_3dLandmarks_from_face_alignment(preds)

# cap = cv2.VideoCapture("./video/test.webm")
# landmark2dtmp=preds[:,0:2]
#
# while True:
#     starttime = datetime.datetime.now()
#
#     _, frame = cap.read()
#     cv2.imshow("frame", frame)
#     frame=cv2.resize(frame,(600,400))
#     dets=camera.detector(frame,1)
#     if len(dets)!=0:
#         shape=camera.predictor(frame,dets[0])
#         i = 0
#         for p in shape.parts():
#             landmark2dtmp[i][0] = p.x
#             landmark2dtmp[i][1] = p.y
#             cv2.circle(frame, (p.x, p.y), 3, (0, 255, 0), -1)
#             i = i + 1
#     endtime = datetime.datetime.now()
#     print((endtime - starttime).microseconds/1000000)
#     cv2.imshow('video',frame)
#     cv2.waitKey(30)

camera.SetCap(w=640,h=360)
landmark2dtmp=preds[:,0:2]
while True:
    starttime = datetime.datetime.now()

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
        landmark2d=landmark2dtmp
        # Calculate Head position under camera(or screen) coordinate with PnP algorithm
        head = gc.Head(landmark3d, landmark2d, frame, 0.1176)
        landmark3dInCam = head.HeadPoseEstimation(camera, screen)

        # Image normalization and get eye region
        M = calculateNormalizationMatrix(head, camera)
        picNorm, landmark2dNorm = imgNormalization(frame, landmark2d, M)
        leye, reye = get_eye_region(picNorm, landmark2dNorm)
        # cv2.imshow('eye',leye)
    endtime=datetime.datetime.now()




    print(frame.shape)
    print((endtime-starttime).seconds)
    print((endtime-starttime).microseconds/1000000)
    cv2.imshow('video',frame)

    cv2.waitKey(1)
