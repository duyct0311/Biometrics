"""
Enrollment.py – Đăng ký khuôn mặt mới vào Database bằng Deep Learning SFace.

Quy trình:
  Camera → MediaPipe landmark → SFace extract_deep_features()
  → Tổng hợp 30 embedding → Trung bình + chuẩn hoá L2 → Lưu Database.csv
"""
import cv2
import time
import os
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import utils


def run_enrollment(options):
    print("\n" + "-" * 45)
    print("  CHẾ ĐỘ ĐĂNG KÝ FACE ID – DEEP LEARNING SFACE")
    print("-" * 45)
    name = input("  Nhập tên người dùng (Enter để huỷ): ").strip()
    if not name:
        print("  Đã huỷ đăng ký.")
        return

    cap = cv2.VideoCapture(0)
    MAX_FRAMES = 30
    embeddings = []
    print(f"\n  [HỆ THỐNG] Camera đang bật... Nhìn thẳng vào camera.")
    print(f"  [HỆ THỐNG] Cần thu thập {MAX_FRAMES} embedding. Nhấn Q để huỷ.\n")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and len(embeddings) < MAX_FRAMES:
            success, frame = cap.read()
            if not success:
                break

            rgb_frame        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image         = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect_for_video(
                mp_image, int(time.time() * 1000)
            )

            display = cv2.flip(frame.copy(), 1)
            h, w    = display.shape[:2]

            if detection_result.face_landmarks:
                lms  = detection_result.face_landmarks[0]
                feat = utils.extract_deep_features(frame, lms)

                if feat is not None:
                    embeddings.append(feat)
                    count      = len(embeddings)
                    progress   = count / MAX_FRAMES
                    bar_len    = 30
                    filled     = int(bar_len * progress)
                    bar        = '█' * filled + '░' * (bar_len - filled)
                    pct        = int(progress * 100)
                    msg        = f"Thu thap Deep Embedding: [{bar}] {pct}%"
                    box_color  = (0, 255, 0)
                else:
                    msg       = "Khuon mat qua xa hoac bi che khuat!"
                    box_color = (0, 165, 255)
            else:
                msg       = "Di chuyen khuon mat vao khung hinh..."
                box_color = (0, 255, 255)

            # Vẽ UI
            overlay        = display.copy()
            cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            cv2.putText(display, f"Dang ky: {name}", (15, 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, msg, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            cv2.imshow("Dang Ky Face ID – Deep Learning (Q: huy)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < MAX_FRAMES:
        print(f"\n  [THẤT BẠI] Chỉ thu được {len(embeddings)}/{MAX_FRAMES} embedding. Đã huỷ.")
        return

    # Tổng hợp: trung bình → chuẩn hoá L2
    final_emb = np.mean(embeddings, axis=0)
    norm      = np.linalg.norm(final_emb)
    if norm > 0:
        final_emb = final_emb / norm

    # Lưu vào Database.csv
    file_exists = os.path.isfile(utils.database_file)
    with open(utils.database_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Header: Name + 128 chiều embedding
            writer.writerow(['Name'] + [f'E{i}' for i in range(len(final_emb))])
        writer.writerow([name] + final_emb.tolist())

    print(f"\n  [THÀNH CÔNG] Đã lưu Deep Embedding 128D của: {name}")
    print(f"  Tổng hồ sơ hiện có: {sum(1 for _ in open(utils.database_file)) - 1}")


if __name__ == '__main__':
    print("Vui lòng chạy file main.py thay vì bật file này trực tiếp.")
