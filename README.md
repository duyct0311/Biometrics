# 🎭 3D Face Biometric Recognition System
### Hệ Thống Định Danh Tái Tạo Mô Hình 3D Khuôn Mặt Từ Ảnh 2D Thông Thường

Dự án này là một hệ thống thị giác máy tính mã nguồn mở, cho phép xác thực sinh trắc học khuôn mặt dưới dạng 3D khối (Depth-Aware) từ một Webcam 2D (RGB) phổ thông. Thuật toán kết hợp sức mạnh của **Deep Learning (Mạng CNN MediaPipe)** để trích xuất 468 điểm tọa độ 3 chiều và **Toán Tỷ Lệ Học** (Euclidean Distance) để lưu trữ Vector Đặc Trưng Sinh Trắc (Biometric Vector).

---

## ✨ Tính năng nổi bật (Key Features)
1. **Trích xuất 3D Point Cloud:** Tái tạo và hiển thị cấu trúc 3D dạng đám mây điểm (Mesh) trực tiếp lơ lửng trên không gian.
2. **Biometric Vector Extraction:** Tự động quy đổi tỷ lệ khuôn mặt của mỗi người thành một Vector Toán học Khối bất biến (Scale Invariant) mà không bị phụ thuộc vào độ xa gần của Camera.
3. **Liveness Detection 3D (Chống giả mạo Bậc 1):** Hệ thống kiểm tra độ lệch chiều sâu (trục Z) giữa chóp mũi và hai má để chặn mọi hình thức dùng Ảnh Thẻ / Màn hình phẳng (Fake 2D Flat Photos).
4. **Blink Detection (Chống giả mạo Bậc 2):** Đo Eye Aspect Ratio (EAR) yêu cầu người dùng nháy mắt để loại bỏ Video/Tượng sáp/Ảnh lừa đảo.
5. **Giao diện Split-Screen Camera:** Tích hợp gọn gàng 3 tính năng vào 1 giao diện điều khiển dòng lệnh trực quan.

---

## 🛠 Yêu cầu Môi trường (Prerequisites)
- **Hệ điều hành:** Windows 10/11, Linux, macOS.
- **LƯU Ý ĐẶC BIỆT DÀNH CHO BẢN WINDOWS:** Toàn bộ đường dẫn thư mục lưu trữ dự án phải KHÔNG có ký tự tiếng Việt (hay bất kỳ ký tự Unicode/Dấu cách đặc biệt nào). Nếu không, module C++ lõi của MediaPipe sẽ văng lỗi `srcdir is not accessible`.
- **Python:** Phiên bản Python `3.8`, `3.9` hoặc `3.10` (Hạn chế xài `3.11` trở lên vì thư viện Tensor C++ cũ dễ vỡ).
- Máy tính bắt buộc phải kết nối với Camera (Webcam).

---

## 🚀 Cài đặt & Cấu hình (Installation)

### Bước 1: Clone dự án tải về máy
```bash
git clone https://github.com/duyct0311/Biometrics.git
cd Biometrics
```

### Bước 2: Cài đặt Thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi động Hệ thống
```bash
python main.py
```
*(Trong lần chạy đầu tiên, mã nguồn sẽ tải xuống mô hình AI `face_landmarker.task` 2MB).*

---

## 🎮 Luồng Vận Hành Hệ Thống (Usage Flow)
Ngay khi khởi động lệnh `main.py`, một Menu sẽ xuất hiện trên Terminal:
- **`[Option 1] Đăng ký Thẻ Face ID:`** Điền tên của bạn. Máy sẽ quét tĩnh 30 khung hình để nén ra Vector Tỷ Lệ hoàn hảo bằng Toán Học Trung Bình Cộng, và lưu vào tệp dữ liệu `Database.csv`.
- **`[Option 2] Cổng An Ninh Chiếm Quyền:`** Màn hình chia đôi (Split-Screen) sẽ hiện lên. Bạn cầm 1 bức ảnh giơ lên, máy sẽ báo cờ Đỏ (Gia Mao). Bạn đưa mặt vào, máy báo Vàng (Pending). Bạn bắt buộc phải **Chớp Mắt** để máy tính qua vòng kiểm duyệt Liveness, sau đó hệ thống sẽ nhận diện tự động và gọi tên bạn Báo Xanh (Verified).
- **`[Option 3] 3D Mesh Viewer:`** Bật độc lập chế độ chiếu ảnh đám mây điểm 3D thô để vọc vạch kỹ thuật chiều sâu điểm Mũi và Má.

---

## 📁 Cấu trúc Mã Nguồn (Architecture)
Dự án được xây dựng gãy gọn theo Tỷ lệ Tiêu Chuẩn Mô-đun (Modular Refactoring):
```text
├── main.py             # Router Điều phối chính của Menu App
├── utils.py            # Trái tim của App: Chứa 100% Thuật toán Toán Học và Liveness Cốt Lõi
├── Enrollment.py       # Module thu thập Vector khung hình
├── Recognition.py      # Module xử lý UI Split-Screen danh tính
├── facemesh.py         # Module Trình chiếu mô phỏng 3D
├── requirements.txt    # Danh sách thư viện tải bằng Pip
└── README.md           # Hướng dẫn này
```
