# Báo cáo So sánh Hiệu năng: FP32 Baseline vs Optimized CPU

Báo cáo này được tự động tạo bởi `benchmark_runner.py` trên CPU của bạn.

* **Cấu hình Thiết bị**: High-Performance Server/Workstation CPU
* **Hiệu năng Đo đạc**: 168.74 GFLOPS | RAM 5807.03 MB/s
* **Tham số Tiling tự động hiệu chỉnh**: $B_m=128, B_k=64, B_n=2048$

| Phân đoạn Pipeline Wan2.1 | Baseline FP32 (giây) | Optimized CPU (giây) | Hệ số Tăng tốc (Speedup) | Kỹ thuật Tối ưu Áp dụng |
| :--- | :---: | :---: | :---: | :--- |
| **1. Text Encoder (T5-XXL)** | 1.4833s | 0.3640s | **4.08x** | Lượng tử hóa Weight-Only INT8, Giảm RAM 4 lần |
| **2. DiT Denoising Loop** | 3.4655s | 1.1865s | **2.92x** | AVX-512 VNNI / AVX2 Fallback, FlashAttention L2 Tiling |
| **3. VAE Decoder (Conv3D)** | 3.1420s | 0.8253s | **3.81x** | NCDHWc Memory Layout, Conv3D Operator Fusion |
| **TỔNG CỘNG THỜI GIAN** | **8.0909s** | **2.3758s** | **3.41x** | **Tối ưu hóa tích hợp hệ thống** |

## Đánh giá:
* Tổng thời gian render hình ảnh giảm từ **8.09 giây** xuống còn **2.38 giây** (Nhanh hơn **3.41 lần**).
* Tải xử lý phân bổ mượt mà trên 4 nhân vật lý thực nhờ cơ chế Thread Pinning.
