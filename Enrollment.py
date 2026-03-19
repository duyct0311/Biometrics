import cv2
import time
import os
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import utils  # Import các hàm toán học chuyên dụng 3D

def run_enrollment(options):
    print("\n" + "-"*35)
    name = input("Nhập tên để Đăng Ký (hoặc enter bỏ qua): ").strip()
    if not name: return
    
    cap = cv2.VideoCapture(0)
    MAX_FRAMES = 30
    vectors = []
    print("\n[HỆ THỐNG] Đang bật Camera... Vui lòng nhìn thẳng.")
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and len(vectors) < MAX_FRAMES:
            success, frame = cap.read()
            if not success: break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            display_frame = cv2.flip(frame.copy(), 1)
            msg, box_color = "Di chuyen khuon mat vao khung hinh", (0, 255, 255)
            
            if detection_result.face_landmarks:
                vec = utils.extract_3d_features(detection_result.face_landmarks[0])
                if vec is not None:
                    vectors.append(vec)
                    msg, box_color = f"Dang ghi nhan Vector sinh trac: {len(vectors)}/{MAX_FRAMES}", (0, 255, 0)
            
            cv2.putText(display_frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            cv2.imshow("HeThong: Dang Ky Mat (Nhan Q huy bo)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if len(vectors) == MAX_FRAMES:
        final_vec = np.mean(vectors, axis=0)
        file_exists = os.path.isfile(utils.database_file)
        with open(utils.database_file, mode='a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(['Name'] + [f'F{i}' for i in range(len(utils.FEATURE_PAIRS))])
            w.writerow([name] + list(final_vec))
        print(f"\n[THÀNH CÔNG] Đã lưu thông tin ID 3D Vector của: {name}")

if __name__ == '__main__':
    print("Vui lòng chạy file main.py thay vì bật file này trực tiếp.")
