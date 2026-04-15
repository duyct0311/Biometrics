# 🎭 3D Face Biometric Recognition System
### Hệ Thống Nhận Diện Sinh Trắc Học Khuôn Mặt – Deep Learning SFace + MediaPipe

Dự án nhận diện khuôn mặt thời gian thực kết hợp hai engine AI:
- **MediaPipe FaceLandmarker** – Trích xuất 468 điểm landmark 3D (x, y, z) từ webcam 2D phổ thông, phục vụ Liveness Detection và Blink Detection.
- **SFace CNN (OpenCV)** – Mạng nơ-ron tích chập tạo embedding vector **128 chiều** (128D), so sánh bằng Cosine Distance để nhận diện danh tính chính xác cao.

---

## ✨ Tính năng nổi bật

| # | Tính năng | Mô tả |
|---|---|---|
| 1 | **Deep Embedding 128D** | SFace CNN alignCrop + feature extraction → vector 128D chuẩn hoá L2 |
| 2 | **Cosine Distance Matching** | So khớp embedding bằng Cosine Distance (ngưỡng 0.40) thay vì Euclidean |
| 3 | **Liveness Detection (Bậc 1)** | Kiểm tra Z-depth delta giữa chóp mũi và gò má → chặn ảnh phẳng / màn hình |
| 4 | **Blink Detection (Bậc 2)** | Đo Eye Aspect Ratio (EAR) yêu cầu chớp mắt → loại video và ảnh chụp |
| 5 | **3D FaceMesh Viewer** | Hiển thị 468 landmark màu theo chiều sâu Z (xanh lá = gần, cyan = xa) |
| 6 | **Split-Screen UI** | Panel trái: camera + nhãn kết quả; Panel phải: 3D mesh real-time |
| 7 | **Auto-download Models** | Tự tải `face_landmarker.task` và `face_recognition_sface_2021dec.onnx` khi khởi động |

---

## 🏗 Kiến trúc hệ thống

```
Frame BGR (Webcam)
       │
       ▼
MediaPipe FaceLandmarker
  → 468 landmarks (x, y, z)
       │
       ├── Liveness Check (z-depth delta)
       ├── Blink Detection (Eye Aspect Ratio)
       │
       ▼
SFace alignCrop (112×112 px)
  → feature() → Embedding 128D (L2-normalized)
       │
       ▼
Cosine Distance vs Database.csv
  → VERIFIED / PENDING / FAKE / UNKNOWN
```

---

## 📁 Cấu trúc Dự Án

```text
biometrics/project/
├── main.py                              # Điều phối menu + tải model tự động
├── utils.py                             # Lõi thuật toán: SFace engine, liveness, database
├── Enrollment.py                        # Thu thập 30 embedding → lưu Database.csv
├── Recognition.py                       # Nhận diện split-screen real-time
├── facemesh.py                          # Trình chiếu 3D Point Cloud
├── requirements.txt                     # Danh sách thư viện pip
├── .gitignore                           # Loại trừ model, venv, dữ liệu nhạy cảm
├── README.md                            # File này
│
├── face_landmarker.task                 # [Tự tải] MediaPipe model (~3.6 MB)
├── face_recognition_sface_2021dec.onnx  # [Tự tải] SFace ONNX model (~37 MB)
└── Database.csv                         # [Tự tạo] Biometric embeddings – NHẠY CẢM
```

> ⚠️ **Lưu ý bảo mật:** `Database.csv`, `*.onnx`, `*.task` được liệt kê trong `.gitignore` và **không được commit** lên repository.

---

## 🛠 Yêu cầu Môi trường

- **Hệ điều hành:** Windows 10/11, Linux, macOS
- **Python:** `3.9` – `3.11`
- **Webcam:** Bắt buộc có camera (USB hoặc tích hợp)
- **⚠️ Windows:** Đường dẫn thư mục project **không được chứa ký tự tiếng Việt có dấu hoặc khoảng trắng đặc biệt** — module C++ lõi của MediaPipe sẽ báo lỗi `srcdir is not accessible`.

---

## 🚀 Cài đặt & Chạy

### Bước 1 – Clone repository
```bash
git clone https://github.com/duyct0311/Biometrics.git
cd Biometrics
```

### Bước 2 – Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Bước 3 – Cài đặt thư viện
```bash
pip install -r requirements.txt
```

> **Lưu ý:** `requirements.txt` chỉ dùng `opencv-contrib-python` (bao gồm module `FaceRecognizerSF`). **Không** cài thêm `opencv-python` vì hai gói xung đột nhau.

### Bước 4 – Khởi động hệ thống
```bash
python main.py
```

**Lần đầu chạy**, hệ thống sẽ tự động tải:
- `face_landmarker.task` (~3.6 MB) từ Google MediaPipe CDN
- `face_recognition_sface_2021dec.onnx` (~37 MB) từ OpenCV Zoo (có resume nếu mất mạng giữa chừng)

---

## 🎮 Hướng dẫn sử dụng

Sau khi khởi động, menu xuất hiện trên Terminal:

```
═══════════════════════════════════════════════════════
  HỆ THỐNG ĐỊNH DANH SINH TRẮC – DEEP LEARNING SFACE
═══════════════════════════════════════════════════════
  1. Đăng ký Face ID mới      (Enroll – SFace 128D)
  2. Nhận diện khuôn mặt      (Liveness + SFace CNN)
  3. Trình chiếu Mô hình 3D   (FaceMesh Viewer)
  4. Thoát phần mềm
═══════════════════════════════════════════════════════
```

### [1] Đăng ký Face ID (Enrollment)
1. Nhập tên người dùng.
2. Camera bật — nhìn thẳng vào camera.
3. Hệ thống thu thập **30 embedding 128D** → tính trung bình → chuẩn hoá L2.
4. Kết quả lưu vào `Database.csv`.

### [2] Nhận diện khuôn mặt (Recognition)
Màn hình chia đôi (Split-Screen):

| Trạng thái | Màu | Ý nghĩa |
|---|---|---|
| `FAKE` | 🔴 Đỏ | Ảnh phẳng / màn hình — Liveness thất bại |
| `PENDING` | 🟡 Vàng | Khuôn mặt thật nhưng chưa chớp mắt |
| `VERIFIED` | 🟢 Xanh | Đã xác thực danh tính thành công |
| `UNKNOWN` | 🟠 Cam | Khuôn mặt thật nhưng không có trong database |

> Thanh thông tin phía trên hiển thị: `Liveness(dZ)` · `Cosine Distance` · `Blink`.

### [3] 3D FaceMesh Viewer
Hiển thị 468 điểm landmark theo màu chiều sâu Z. Nhấn **Q** để thoát.

---

## ⚙️ Thư viện sử dụng (`requirements.txt`)

| Gói | Phiên bản | Mục đích |
|---|---|---|
| `opencv-contrib-python` | `>=4.7.0` | Xử lý ảnh + `cv2.FaceRecognizerSF` (SFace engine) |
| `mediapipe` | `>=0.10.0` | FaceLandmarker Tasks API (468 landmarks 3D) |
| `numpy` | `>=1.21.0` | Tính toán vector, ma trận, Cosine Distance |

---

## 🔒 Bảo mật & Quyền riêng tư

- **`Database.csv`** chứa embedding sinh trắc học cá nhân — được liệt kê trong `.gitignore`, **không bao giờ commit lên repository công khai**.
- **Model files** (`*.onnx`, `*.task`) là file nhị phân lớn — cũng được loại trừ qua `.gitignore`; hãy để `main.py` tải tự động hoặc dùng **Git LFS** nếu cần phân phối trong team.

---

## 🧠 Thuật toán cốt lõi

### SFace Face Recognition (utils.py)
```
1. Lấy 5 điểm mốc [468,473,1,61,291] từ MediaPipe landmarks
2. Tính bounding box + padding 20%
3. rec.alignCrop(frame, bbox)  →  ảnh khuôn mặt chuẩn 112×112 px
4. rec.feature(aligned_face)   →  vector 128D chuẩn hoá L2
5. Cosine Distance = 1 - dot(a,b)/(||a||·||b||)
6. Distance < 0.40  →  MATCH
```

### Liveness Detection (utils.py)
```
depth = |avg_cheek_z - nose_z|
depth < 0.045  →  FAKE (ảnh phẳng không có chiều sâu 3D)
```

### Blink Detection – EAR (utils.py)
```
EAR = (dist(159,145) + dist(386,374)) / (2 × dist(33,133))
EAR < 0.23  →  Đang chớp mắt → has_blinked = True
```
