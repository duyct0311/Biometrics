"""
utils.py – Thư viện lõi hệ thống nhận diện khuôn mặt Deep Learning SFace.

Luồng nhận diện:
  Frame BGR  →  MediaPipe Landmarks  →  extract_deep_features()
             →  SFace alignCrop + CNN  →  Embedding 128D
             →  compare_faces()  →  Cosine Distance  →  Danh tính
"""
import csv
import os
import math
import numpy as np
import cv2

# ──────────────────────────────────────────────────────────────────────────
# Hằng số
# ──────────────────────────────────────────────────────────────────────────
database_file = 'Database.csv'

# Ngưỡng Cosine Distance: 0 = giống hệt, 1 = hoàn toàn khác
# Nghiên cứu gốc khuyến nghị 0.363; ta dùng 0.40 để an toàn hơn
SFACE_COSINE_THRESHOLD = 0.40

# Liveness & Blink (dùng MediaPipe landmarks)
LIVENESS_THRESHOLD = 0.045   # Z-depth delta nhỏ → ảnh phẳng (giả)
EAR_THRESHOLD      = 0.23    # Eye Aspect Ratio nhỏ → đang chớp mắt

# 5 điểm mốc MediaPipe cần cho SFace alignCrop
# Thứ tự: [MắtTrái, MắtPhải, ĐỉnhMũi, KhóeMiệngTrái, KhóeMiệngPhải]
MEDIAPIPE_5PTS = [468, 473, 1, 61, 291]

# ──────────────────────────────────────────────────────────────────────────
# Singleton SFace Recognizer
# ──────────────────────────────────────────────────────────────────────────
_recognizer = None


def load_sf_recognizer(model_path: str = 'face_recognition_sface_2021dec.onnx'):
    """
    Nạp (hoặc trả lại) engine SFace.
    Gọi một lần duy nhất từ main.py khi khởi động.
    """
    global _recognizer
    if _recognizer is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Không tìm thấy model SFace: {model_path}\n"
                "Hãy chạy main.py để tải tự động, hoặc tải thủ công."
            )
        _recognizer = cv2.FaceRecognizerSF.create(model_path, "")
    return _recognizer


# ──────────────────────────────────────────────────────────────────────────
# Trích xuất Deep Embedding 128D từ SFace CNN
# ──────────────────────────────────────────────────────────────────────────
def extract_deep_features(bgr_frame: np.ndarray, landmarks) -> np.ndarray | None:
    """
    Trích xuất vector Embedding 128D từ một khuôn mặt trong frame.

    Quy trình:
      1. Lấy 5 điểm mốc pixel (px, py) từ MediaPipe landmarks (tỉ lệ → pixel).
      2. Tính bbox xấp xỉ có thêm 20% lề mỗi chiều.
      3. SFace alignCrop() → cắt và thẳng hướng khuôn mặt chuẩn 112×112.
      4. SFace feature()  → vector 128D chuẩn hoá L2.
    Trả về numpy array (128,) hoặc None nếu thất bại.
    """
    rec = load_sf_recognizer()
    h, w = bgr_frame.shape[:2]

    # Lấy toạ độ pixel của 5 điểm mốc
    pts_px = []
    for idx in MEDIAPIPE_5PTS:
        lm = landmarks[idx]
        pts_px.append([lm.x * w, lm.y * h])
    pts_px = np.array(pts_px, dtype=np.float32)

    # Tính bbox (x, y, width, height) với lề 20%
    xs, ys = pts_px[:, 0], pts_px[:, 1]
    face_w = xs.max() - xs.min()
    face_h = ys.max() - ys.min()
    pad_x  = face_w * 0.20
    pad_y  = face_h * 0.20
    x_min  = max(0.0, xs.min() - pad_x)
    y_min  = max(0.0, ys.min() - pad_y)
    x_max  = min(float(w), xs.max() + pad_x)
    y_max  = min(float(h), ys.max() + pad_y)
    bbox   = np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=np.float32)

    # Căn chỉnh + trích đặc trưng
    try:
        aligned_face = rec.alignCrop(bgr_frame, bbox)
        feature      = rec.feature(aligned_face)
        return feature.flatten().astype(np.float32)
    except Exception as e:
        print(f"[WARN] extract_deep_features: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────
# So khớp hai embedding
# ──────────────────────────────────────────────────────────────────────────
def compare_faces(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Tính Cosine Distance giữa hai embedding 128D bằng NumPy thuần.
    Không dùng cv2.FaceRecognizerSF.FR_COSINE vì OpenCV 4.13 bỏ hằng số này.

    Cosine Similarity = dot(a, b) / (||a|| * ||b||)
    Cosine Distance   = 1 - Cosine Similarity
    Giá trị: 0 = giống hệt, 1 = hoàn toàn khác.
    Ngưỡng chấp nhận: < SFACE_COSINE_THRESHOLD (0.40)
    """
    n1 = np.linalg.norm(feat1)
    n2 = np.linalg.norm(feat2)
    if n1 == 0 or n2 == 0:
        return 1.0                        # không thể so sánh → coi như khác hoàn toàn
    similarity = np.dot(feat1, feat2) / (n1 * n2)
    # clip để tránh lỗi float khi similarity vượt [-1, 1] do làm tròn
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return 1.0 - similarity               # đổi sang distance (thấp = giống)


# ──────────────────────────────────────────────────────────────────────────
# Liveness & Blink detection (dùng MediaPipe – giữ nguyên từ phiên bản cũ)
# ──────────────────────────────────────────────────────────────────────────
def _dist3d(lm1, lm2) -> float:
    return math.sqrt(
        (lm1.x - lm2.x) ** 2 +
        (lm1.y - lm2.y) ** 2 +
        (lm1.z - lm2.z) ** 2
    )


def get_ear(landmarks) -> float:
    """Eye Aspect Ratio – phát hiện chớp mắt bằng tỉ lệ dọc/ngang mắt."""
    ear_l = _dist3d(landmarks[159], landmarks[145]) / _dist3d(landmarks[33],  landmarks[133])
    ear_r = _dist3d(landmarks[386], landmarks[374]) / _dist3d(landmarks[362], landmarks[263])
    return (ear_l + ear_r) / 2.0


def check_liveness(landmarks) -> float:
    """
    Tính Z-Depth delta giữa gò má và mũi.
    Ảnh in phẳng/màn hình → delta ≈ 0 (không có chiều sâu 3D).
    Khuôn mặt thật → delta lớn hơn.
    """
    avg_cheek_z = (landmarks[234].z + landmarks[454].z) / 2.0
    return abs(avg_cheek_z - landmarks[1].z)


# ──────────────────────────────────────────────────────────────────────────
# Cơ sở dữ liệu CSV
# ──────────────────────────────────────────────────────────────────────────
def load_database() -> tuple[list[str], list[np.ndarray]]:
    """
    Đọc Database.csv.
    Trả về (danh_sach_ten, danh_sach_embedding_128D).
    """
    names, vectors = [], []
    if os.path.exists(database_file):
        with open(database_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)          # bỏ dòng header
            for row in reader:
                if len(row) > 1:
                    names.append(row[0])
                    vectors.append(
                        np.array([float(x) for x in row[1:]], dtype=np.float32)
                    )
    return names, vectors
