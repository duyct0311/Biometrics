import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import Enrollment
import Recognition
import facemesh

model_path = 'face_landmarker.task'

def main():
    # Tải model AI Google
    if not os.path.exists(model_path):
        print("\n[AI MODULE] Đang tải mô hình phân tích mốc sinh trắc, xin chờ...")
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", model_path)
        
    # Nạp Model vào Engine
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO, 
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Bộ định tuyến (Menu Router)
    while True:
        print("\n" + "="*50)
        print(" THIẾT KẾ MÔ-ĐUN: HỆ THỐNG GƯƠNG MẶT 3D")
        print("="*50)
        print("  1. Đăng ký Thẻ Face ID Mới (Enroll)")
        print("  2. Chạy Kiểm Tra An Ninh (Nhận Diện + Liveness)")
        print("  3. Trình chiếu Mô hình 3D (Soi Gương 3D)")
        print("  4. Thoát phần mềm\n")
        
        choice = input(" Vui lòng chọn (1/2/3/4): ").strip()
        
        if choice == '1':
            Enrollment.run_enrollment(options)
        elif choice == '2':
            Recognition.run_recognition(options)
        elif choice == '3':
            facemesh.run_facemesh(options)
        elif choice == '4':
            print("Đã thoát ứng dụng. Demo kết thúc!")
            break
        else:
            print("Lựa chọn không hợp lệ, Xin vui lòng gõ (1-4)!")

if __name__ == '__main__':
    main()
