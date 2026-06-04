# Báo cáo So sánh Hiệu năng: FP32 Baseline vs Optimized CPU

Báo cáo này được tự động tạo bởi `benchmark_runner.py` trên CPU của bạn.

* **Cấu hình Thiết bị**: High-Performance Server/Workstation CPU
* **Hiệu năng Đo đạc**: 168.74 GFLOPS | RAM 5807.03 MB/s
* **Tham số Tiling tự động hiệu chỉnh**: $B_m=128, B_k=64, B_n=2048$

| Phân đoạn Pipeline Wan2.1 | Baseline FP32 (giây) | Optimized CPU (giây) | Hệ số Tăng tốc (Speedup) | Kỹ thuật Tối ưu Áp dụng |
| :--- | :---: | :---: | :---: | :--- |
| **1. Text Encoder (T5-XXL)** | 1.3010s | 0.3498s | **3.72x** | Lượng tử hóa Weight-Only INT8, Giảm RAM 4 lần |
| **2. DiT Denoising Loop** | 3.2063s | 1.1395s | **2.81x** | AVX-512 VNNI / AVX2 Fallback, FlashAttention L2 Tiling |
| **3. VAE Decoder (Conv3D)** | 2.7042s | 0.7996s | **3.38x** | NCDHWc Memory Layout, Conv3D Operator Fusion |
| **TỔNG CỘNG THỜI GIAN** | **7.2114s** | **2.2889s** | **3.15x** | **Tối ưu hóa tích hợp hệ thống** |

## Đánh giá:
* Tổng thời gian render hình ảnh giảm từ **7.21 giây** xuống còn **2.29 giây** (Nhanh hơn **3.15 lần**).
* Tải xử lý phân bổ mượt mà trên 4 nhân vật lý thực nhờ cơ chế Thread Pinning.
