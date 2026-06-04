# Kế hoạch Nâng cấp và Tối ưu hóa: Wan2.1 3D-VAE & DiT on CPU

Tài liệu này trình bày chi tiết kế hoạch tối ưu hóa cho hai thành phần quan trọng nhất của pipeline sinh video/hình ảnh **Wan2.1 (wan2gp)**: Bộ tự động mã hóa biến phân 3 chiều (3D-VAE) và Khối biến đổi khuếch tán không-thời gian (Spatial-Temporal DiT).

---

## 1. Tối ưu hóa 3D-VAE Decoder

### Vấn đề:
VAE Decoder của Wan2.1 chuyển đổi latent representation (không gian tiềm ẩn) trở lại không gian pixel của hình ảnh/video. Nó sử dụng phép **Tích chập Causal 3D (Causal 3D Convolutions)**. Phép toán này yêu cầu:
* Lượng tính toán gấp nhiều lần so với Conv2D truyền thống do có thêm chiều thời gian (hoặc số khung hình).
* Lưu lượng truy cập bộ nhớ cực kỳ lớn để lưu trữ các bộ đệm khung hình trung gian.

### Giải pháp tối ưu:
1. **Layout Conversion (NCHWc)**:
   * Chuyển đổi tensor 3D VAE từ định dạng mặc định sang định dạng tối ưu cho vector hóa: `NCDHW` $\rightarrow$ `NCDHWc` (với $c = 16$ cho AVX-512 hoặc $c = 8$ cho AVX2).
   * Điều này đảm bảo toàn bộ phép tích chập dọc theo chiều kênh (Channel) được thực thi bằng các lệnh SIMD song song mà không bị phân tán bộ nhớ.
2. **Channel-Last Conv3D**:
   * Định dạng Channels-Last (`NDHWC`) giúp tăng Spatial Locality khi thực hiện phép nhân-cộng chập.
3. **AVX-512 / AVX2 VNNI Fusion**:
   * Áp dụng phép chiếu tích chập 3D trực tiếp bằng lượng tử hóa INT8. Các trọng số tích chập của VAE được chuyển sang định dạng INT8 cố định từ trước. Activation được lượng tử hóa động theo từng block ảnh.

---

## 2. Tối ưu hóa Spatial-Temporal DiT Block

### Vấn đề:
DiT Block trong Wan2.1 chịu trách nhiệm khử nhiễu (denoising). Nó bao gồm các khối **3D Self-Attention (Spatial-Temporal Attention)** và **FFN**.
* Attention Score yêu cầu tính toán $QK^T$ và nhân với $V$. Khi số lượng patch tăng lên (do ảnh độ phân giải cao hoặc video dài), ma trận Attention phình to rất nhanh.
* Các lớp Linear Projection trong Attention và FFN thực hiện các phép nhân ma trận (MatMul) khổng lồ liên tục.

### Giải pháp tối ưu:
1. **FlashAttention CPU-Equivalent (Tiling Attention)**:
   * Do CPU không có bộ nhớ shared tốc độ cao giống GPU nhưng có L2/L3 Cache lớn, chúng ta chia khối (Tile) ma trận $Q, K, V$ thành các khối con kích thước nhỏ vừa khít Cache L2.
   * Tính toán Attention từng khối con và dồn tích lũy trực tiếp (online softmax), tránh việc cấp phát RAM cho ma trận Attention Score đầy đủ ($S \times S$). Điều này tiết kiệm $O(S^2)$ bộ nhớ và loại bỏ nghẽn băng thông RAM.
2. **SmoothQuant cho DiT Linear Layers**:
   * LLM và DiT thường xuất hiện Outliers trong Activation. Áp dụng SmoothQuant để chia nhỏ độ khó lượng tử hóa giữa Activation và Weights trước khi đưa vào các layer FFN.
   * Đảm bảo chuyển đổi hoàn toàn các phép toán MatMul của DiT sang INT8 dùng tập lệnh VNNI phần cứng.

---

## 3. Tối ưu hóa Text Encoder (T5-XXL)

### Vấn đề:
T5-XXL có kích thước khoảng 4.3B tham số, chiếm gần 16GB RAM nếu chạy ở FP32. Trên CPU laptop thông thường, việc load mô hình này dễ gây tràn RAM và treo máy.

### Giải pháp tối ưu:
* **INT8 / INT4 Weight-Only Quantization**:
   * Do Text Encoder chỉ chạy một lần ở đầu pipeline (trước vòng lặp khuếch tán), tốc độ không cần quá cực hạn nhưng yêu cầu dung lượng bộ nhớ cực thấp.
   * Nén trọng số T5-XXL xuống INT8 (hoặc INT4). Khi inference, trọng số được giải nén động (dequantize) sang FP16/FP32 ngay trên thanh ghi SIMD trước khi nhân với activation. Điều này giảm dung lượng RAM từ 16GB xuống còn dưới 4.5GB.
