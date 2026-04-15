import os
import sys
import time
import urllib.request
import urllib.error

# Buộc stdout dùng UTF-8 để tránh UnicodeEncodeError trên Windows (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import Enrollment
import Recognition
import facemesh
import utils

# ── Đường dẫn model ────────────────────────────────────────────────────────
MODEL_MP   = 'face_landmarker.task'
MODEL_SF   = 'face_recognition_sface_2021dec.onnx'
SF_MIN_MB  = 30            # File hợp lệ phải ≥ 30 MB
CHUNK      = 64 * 1024     # 64 KB / chunk

# Nhiều mirror để fallback nếu 1 link chết
SF_URLS = [
    # Mirror 1: media.githubusercontent.com (GitHub LFS CDN)
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    # Mirror 2: raw GitHub (cho file nhỏ, thường bị redirect về LFS)
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]


# ── Bộ tải mạnh mẽ: Resume + Retry + Progress ─────────────────────────────
def _download(url: str, dest: str, min_size_mb: int = SF_MIN_MB) -> bool:
    """
    Tải file với:
      - HTTP Range header để resume nếu tải dở.
      - Đọc từng chunk 64 KB, không bị timeout RAM.
      - Hiển thị tiến trình (X.X / Y.Y MB).
      - Retry tự động lên đến 5 lần với delay tăng dần.
    Trả về True nếu thành công.
    """
    MAX_RETRY = 5
    min_bytes = min_size_mb * 1024 * 1024

    for attempt in range(1, MAX_RETRY + 1):
        try:
            # Resume: gửi Range header nếu file đang tải dở
            existing = os.path.getsize(dest) if os.path.exists(dest) else 0
            headers  = {}
            if existing > 0:
                headers['Range'] = f'bytes={existing}-'
                print(f"  [RESUME] Tiếp tục từ {existing/1024/1024:.1f} MB...")

            req  = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=60)

            # Tổng kích thước
            cl = resp.headers.get('Content-Length') or resp.headers.get('content-length')
            total = (int(cl) + existing) if cl else 0

            mode = 'ab' if existing > 0 else 'wb'
            downloaded = existing

            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        mb_now = downloaded / 1024 / 1024
                        mb_tot = total / 1024 / 1024
                        print(f"\r  [{bar}] {mb_now:.1f}/{mb_tot:.1f} MB ({pct:.0f}%)",
                              end='', flush=True)
                    else:
                        print(f"\r  Đã tải: {downloaded/1024/1024:.1f} MB",
                              end='', flush=True)
            print()  # xuống dòng sau progress bar

            # Kiểm tra kích thước
            actual = os.path.getsize(dest)
            if actual >= min_bytes:
                print(f"  ✓ Tải hoàn tất: {actual/1024/1024:.1f} MB")
                return True
            else:
                print(f"  [WARN] File chỉ {actual/1024/1024:.1f} MB (cần ≥ {min_size_mb} MB). "
                      "Thử lại...")
                # Không xoá → lần sau resume tiếp

        except KeyboardInterrupt:
            print("\n  [INFO] Đã dừng. File dở sẽ được tiếp tục lần chạy sau.")
            sys.exit(0)

        except Exception as e:
            print(f"\n  [WARN] Lần {attempt}/{MAX_RETRY}: {e}")
            if attempt < MAX_RETRY:
                wait = min(2 ** attempt, 30)
                print(f"  Thử lại sau {wait}s...")
                time.sleep(wait)

    return False


def _ensure_model(dest: str, urls: list[str], label: str,
                  min_size_mb: int = SF_MIN_MB) -> bool:
    """Kiểm tra file – nếu thiếu hoặc nhỏ hơn min, tải từ danh sách URL."""
    min_bytes = min_size_mb * 1024 / 1024 * 1024 * 1024 / 1024  # tránh nhầm
    min_bytes = min_size_mb * 1024 * 1024

    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return True  # File OK rồi

    if os.path.exists(dest):
        print(f"[AI MODULE] File {dest} bị tải dở ({os.path.getsize(dest)/1024/1024:.1f} MB),"
              " tiếp tục tải...")

    for i, url in enumerate(urls, 1):
        print(f"\n[AI MODULE] Đang tải {label} – mirror {i}/{len(urls)}...")
        print(f"  URL: {url[:70]}...")
        if _download(url, dest, min_size_mb):
            return True
        print(f"  Mirror {i} thất bại, thử mirror tiếp theo...")

    return False


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  HỆ THỐNG ĐỊNH DANH SINH TRẮC – DEEP LEARNING SFACE  ")
    print("=" * 55)

    # 1. MediaPipe FaceLandmarker
    if not os.path.exists(MODEL_MP):
        print("\n[AI MODULE] Tải MediaPipe FaceLandmarker...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            MODEL_MP
        )
        print("[AI MODULE] ✓ MediaPipe sẵn sàng.")

    # 2. SFace ONNX (~36 MB) – với downloader mạnh mẽ
    if not _ensure_model(MODEL_SF, SF_URLS, "SFace ONNX (~36 MB)"):
        print("\n[LỖI] Không thể tải model SFace sau nhiều lần thử.")
        print("  → Hãy tải thủ công tại:")
        print("    https://github.com/opencv/opencv_zoo/raw/main/"
              "models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
        print(f"  → Đặt vào thư mục: {os.path.abspath(MODEL_SF)}")
        sys.exit(1)

    # 3. Nạp SFace Singleton
    print("\n[AI MODULE] Đang khởi tạo engine Deep Learning SFace...")
    utils.load_sf_recognizer(MODEL_SF)
    print("[AI MODULE] ✓ SFace Engine sẵn sàng.\n")

    # 4. Cấu hình MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_MP)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 5. Menu
    while True:
        print("\n" + "=" * 55)
        print("  1. Đăng ký Face ID mới      (Enroll – SFace 128D)")
        print("  2. Nhận diện khuôn mặt      (Liveness + SFace CNN)")
        print("  3. Trình chiếu Mô hình 3D   (FaceMesh Viewer)")
        print("  4. Thoát phần mềm")
        print("=" * 55)

        choice = input("  Vui lòng chọn (1-4): ").strip()
        if choice == '1':
            Enrollment.run_enrollment(options)
        elif choice == '2':
            Recognition.run_recognition(options)
        elif choice == '3':
            facemesh.run_facemesh(options)
        elif choice == '4':
            print("Đã thoát. Tạm biệt!")
            break
        else:
            print("  [!] Lựa chọn không hợp lệ (1–4).")


if __name__ == '__main__':
    main()
