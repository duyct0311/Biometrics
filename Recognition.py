import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import utils

def run_recognition(options):
    names, vectors = utils.load_database()
    if not names:
        print("\n[LỖI CSDL] Database trống! Hãy chạy Đăng Ký trước (Bấm 1).")
        return
        
    print(f"\n[HỆ THỐNG CHIẾN ĐẤU] Đã Kích Hoạt Liveness Cấp 2. Database ({len(names)} hồ sơ).")
    cap = cv2.VideoCapture(0)
    has_blinked = False
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            # Khởi tạo 2 màn hình
            display_frame = cv2.flip(frame.copy(), 1)
            black_bg = np.zeros((h, w, 3), dtype=np.uint8) # Màn hình đen cùng kích cỡ
            
            # Tiêu đề màn hình đen
            cv2.putText(black_bg, "3D FACE MESH VIEWER", (w//2 - 120, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            
            if detection_result.face_landmarks:
                lms = detection_result.face_landmarks[0]
                
                # ==== VẼ MÔ HÌNH MẠNG LƯỚI 3D (Bên Phải) ====
                for landmark in lms:
                    # Lật x (1.0 - x) để ảnh 3D phản chiếu y hệt màn hình Camera (Soi gương)
                    mesh_x = int((1.0 - landmark.x) * w)
                    mesh_y = int(landmark.y * h)
                    # Chỉnh màu độ sâu z
                    color, r = ((0, 255, 0), 1) if landmark.z >= -0.05 else ((0, 255, 255), 2)
                    cv2.circle(black_bg, (mesh_x, mesh_y), r, color, -1)
                
                # ==== LOGIC ĐỊNH DANH (Bên Trái) ====
                x = int((1.0 - lms[10].x) * w)
                y = int(lms[10].y * h) - 20
                
                depth = utils.check_liveness(lms)
                ear = utils.get_ear(lms)
                if ear < utils.EAR_THRESHOLD:
                    has_blinked = True
                
                vec = utils.extract_3d_features(lms)
                best_name = "Unknown"
                min_distance = float('inf')
                
                if vec is not None:
                    distances = [np.linalg.norm(vec - db_v) for db_v in vectors]
                    min_distance = min(distances)
                    if min_distance < utils.THRESHOLD:
                        best_name = names[distances.index(min_distance)]
                
                color = (255, 255, 255)
                box_color = (0, 0, 0)
                status_text = ""
                
                if depth < utils.LIVENESS_THRESHOLD:
                    status_text = f"FAKE: {best_name}"
                    color, box_color = (0, 0, 255), (0, 0, 255)
                    cv2.rectangle(display_frame, (0, 0), (w, h), box_color, 8)
                    cv2.putText(display_frame, status_text, (max(0, x-50), max(20, y)), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
                elif not has_blinked:
                    status_text = f"PENDING: {best_name}"
                    color, box_color = (0, 255, 255), (0, 255, 255)
                    cv2.putText(display_frame, f"CHOP MAT DE XAC NHAN!", (30, h-30), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                    cv2.putText(display_frame, status_text, (max(0, x-60), max(20, y)), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
                else:
                    if best_name != "Unknown":
                        status_text = f"VERIFIED: {best_name}"
                        color, box_color = (0, 255, 0), (0, 255, 0)
                    else:
                        status_text = f"UNKNOWN FACE"
                        color, box_color = (0, 165, 255), (0, 165, 255)
                    cv2.rectangle(display_frame, (0, 0), (w, h), box_color, 4)
                    cv2.putText(display_frame, status_text, (max(0, x-60), max(20, y)), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

                cv2.putText(display_frame, f"Depth: {depth:.3f} | Dist: {min_distance:.3f} | Blink: {has_blinked}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # Gộp 2 khung hình ngang nhau (Side by side Array hstack)
            combined_frame = np.hstack((display_frame, black_bg))
            cv2.imshow("He Thong Nhan Dien: Giao dien Chuyen Nghiep (Nhan Q thoat)", combined_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Vui lòng chạy file main.py thay vì bật file này trực tiếp.")
