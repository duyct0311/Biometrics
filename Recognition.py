"""
Recognition.py – Nhận diện khuôn mặt thời gian thực
  bằng Deep Learning SFace + Liveness Check + Blink Detection.

Màn hình Split-Screen:
  Trái: Camera thực + overlay nhãn (VERIFIED / PENDING / FAKE / UNKNOWN)
  Phải: 3D FaceMesh Viewer (màu theo độ sâu Z)
"""
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import utils


def run_recognition(options):
    names, db_embeddings = utils.load_database()
    if not names:
        print("\n  [LỖI] Database trống! Hãy chạy Đăng Ký trước (Bấm 1).")
        return

    print(f"\n  [HỆ THỐNG] Đã nạp {len(names)} hồ sơ.")
    print(f"  [HỆ THỐNG] Ngưỡng Cosine Distance: {utils.SFACE_COSINE_THRESHOLD}")
    print("  [HỆ THỐNG] Nhấn Q để thoát.\n")

    cap = cv2.VideoCapture(0)

    # State Machine
    has_blinked    = False
    last_face_time = time.time()

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            h, w, _          = frame.shape
            rgb_frame        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image         = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect_for_video(
                mp_image, int(time.time() * 1000)
            )

            # Hai panel hiển thị
            display  = cv2.flip(frame.copy(), 1)
            mesh_bg  = np.zeros((h, w, 3), dtype=np.uint8)

            # Tiêu đề panel phải
            cv2.putText(mesh_bg, "3D FACE MESH VIEWER",
                        (w // 2 - 130, 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

            if detection_result.face_landmarks:
                last_face_time = time.time()
                lms = detection_result.face_landmarks[0]

                # ── Vẽ 3D FaceMesh (panel phải) ─────────────────────────
                for lm in lms:
                    mx = int((1.0 - lm.x) * w)
                    my = int(lm.y * h)
                    # Màu theo chiều sâu: gần = xanh lá, xa = cyan
                    color = (0, 255, 0) if lm.z >= -0.05 else (0, 255, 255)
                    cv2.circle(mesh_bg, (mx, my), 1, color, -1)

                # ── 1. LIVENESS CHECK & BLINK CHECK ─────────────────────
                depth = utils.check_liveness(lms)
                ear   = utils.get_ear(lms)
                if ear < utils.EAR_THRESHOLD:
                    has_blinked = True

                # ── 2. DEEP LEARNING NHẬN DIỆN (SFace) ──────────────────
                best_name     = "Unknown"
                best_distance = float('inf')

                feat = utils.extract_deep_features(frame, lms)
                if feat is not None and db_embeddings:
                    distances     = [utils.compare_faces(feat, db_v)
                                     for db_v in db_embeddings]
                    best_distance = min(distances)
                    if best_distance < utils.SFACE_COSINE_THRESHOLD:
                        best_name = names[distances.index(best_distance)]

                # Vị trí hiển thị nhãn (đã lật X)
                label_x = max(0,  int((1.0 - lms[10].x) * w) - 70)
                label_y = max(30, int(lms[10].y * h) - 25)

                # ── 3. PHÂN NHÁNH KẾT QUẢ ───────────────────────────────
                if depth < utils.LIVENESS_THRESHOLD:
                    # ❌ Ảnh giả / màn hình / in ấn
                    status    = f"FAKE: {best_name}"
                    color     = (0, 0, 255)
                    cv2.rectangle(display, (0, 0), (w, h), color, 10)
                    cv2.putText(display, status, (label_x, label_y),
                                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 3)

                elif not has_blinked:
                    # ⏳ Chờ chớp mắt xác nhận
                    status = f"PENDING: {best_name}"
                    color  = (0, 220, 220)
                    cv2.putText(display, ">> CHOP MAT DE XAC NHAN <<",
                                (w // 2 - 210, h - 35),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                    cv2.putText(display, status, (label_x, label_y),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

                else:
                    if best_name != "Unknown":
                        # ✅ Danh tính xác thực thành công
                        status = f"VERIFIED: {best_name}"
                        color  = (0, 255, 0)
                    else:
                        # ❓ Không nhận ra
                        status = "UNKNOWN FACE"
                        color  = (0, 140, 255)
                    cv2.rectangle(display, (0, 0), (w, h), color, 5)
                    cv2.putText(display, status, (label_x, label_y),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

                # ── Thanh thông tin chi tiết (phía trên) ────────────────
                dist_str = f"{best_distance:.3f}" if best_distance != float('inf') else "N/A"
                info = (f"Liveness(dZ):{depth:.3f}  "
                        f"Cosine:{dist_str}  "
                        f"Blink:{has_blinked}")
                cv2.rectangle(display, (0, 0), (w, 45), (0, 0, 0), -1)
                cv2.putText(display, info, (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

            else:
                # Không có mặt → reset blink sau 1.5s
                if time.time() - last_face_time > 1.5:
                    has_blinked = False
                cv2.putText(display, "DANG TIM KIEM KHUON MAT...",
                            (w // 2 - 200, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

            combined = np.hstack((display, mesh_bg))
            cv2.imshow("Nhan Dien: Deep Learning SFace  (Q: thoat)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Vui lòng chạy file main.py thay vì bật file này trực tiếp.")
