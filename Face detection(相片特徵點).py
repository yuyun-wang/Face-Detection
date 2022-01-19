import dlib
import cv2
 
#Dlib facial landmarks model的path
predictor_path = "model/shape_predictor_68_face_landmarks.dat"
#待處理的相片
face_path = "face1.jpg"
 
#於landmarks上畫圓，標識特徴點
def renderFace(im, landmarks, color=(0, 255, 0), radius=3):
  for p in landmarks.parts():
    cv2.circle(im, (p.x, p.y), radius, color, -1)
 
#detector為臉孔偵測，predictor為landmarks偵測
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
 
#讀入相片
img = cv2.imread(face_path)
#偵測臉孔
dets = detector(img, 1)
 
#針對相片中的每張臉孔偵測五個landmarks
for k, d in enumerate(dets):
    shape = predictor(img, d)
    renderFace(img, shape)
 
cv2.imshow("face-rendered", img)
cv2.waitKey(0)
