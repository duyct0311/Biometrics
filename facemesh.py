import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import vision

def run_facemesh(options):
    print("\n[HỆ THỐNG] Đang bật chế độ Quét 3D Point Cloud...")
    cap = cv2.VideoCapture(0)
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, int(time.time() * 1000))
            
            disp = frame.copy()
            if res.face_landmarks:
                for face_landmarks in res.face_landmarks:
                    for landmark in face_landmarks:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        color, r = ((0,255,0), 1) if landmark.z >= -0.05 else ((0,255,255), 2)
                        cv2.circle(disp, (x, y), r, color, -1)
                        
            disp = cv2.flip(disp, 1)
            cv2.putText(disp, "3D FACE MESH VIEWER", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.imshow("System: 3D Face Model (Nhan Q thoat)", disp)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
