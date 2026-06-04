# Giải thuật Loop Tiling & Operator Fusion

Tài liệu này đặc tả các thuật toán xử lý cấp thấp được triển khai để tối ưu hóa việc sử dụng Cache CPU (L1/L2/L3) và giảm thiểu băng thông truy cập RAM khi chạy mô hình **Wan2.1**.

---

## 1. Thuật toán Loop Tiling (Chia khối ma trận)

Trong phép nhân ma trận $C = A \times B$ với $A \in \mathbb{R}^{M \times K}$ và $B \in \mathbb{R}^{K \times N}$, vòng lặp lồng nhau thông thường gây ra nhiều Cache Miss ở cột của $B$. Loop Tiling chia nhỏ các chiều $M, N, K$ thành các khối con kích thước $B_m, B_n, B_k$.

### Mã giả thuật toán (Tiled MatMul):

```python
# M, N, K: Kích thước ma trận gốc
# B_m, B_n, B_k: Kích thước khối (Tile Size) được chọn dựa trên dung lượng L2 Cache
# Kích thước khối được tính toán sao cho: (B_m * B_k + B_k * B_n) * sizeof(float) <= L2_Cache_Size

def tiled_matmul(A, B, C, M, N, K, B_m, B_n, B_k):
    # Vòng lặp ngoài cùng nhảy theo kích thước khối
    for ii in range(0, M, B_m):
        for jj in range(0, N, B_n):
            for kk in range(0, K, B_k):
                
                # Tính toán giới hạn biên cho khối hiện tại (xử lý phần dư)
                i_end = min(ii + B_m, M)
                j_end = min(jj + B_n, N)
                k_end = min(kk + B_k, K)
                
                # Thực hiện nhân ma trận trên các khối nhỏ (Micro-kernel)
                # Toàn bộ khối con A[ii:i_end, kk:k_end] và B[kk:k_end, jj:j_end] nằm gọn trong L2 Cache
                for i in range(ii, i_end):
                    for j in range(jj, j_end):
                        sum_val = 0.0
                        for k in range(kk, k_end):
                            sum_val += A[i, k] * B[k, j]
                        C[i, j] += sum_val
```

---

## 2. Giải thuật Operator Fusion (Hợp nhất Toán tử)

Hợp nhất các toán tử toán học nhỏ liền kề để loại bỏ bước ghi kết quả trung gian về RAM. 

### Ví dụ: Hợp nhất phép nhân ma trận (MatMul), cộng Bias (BiasAdd) và kích hoạt (GELU)

#### Quy trình khi KHÔNG Fusion:
1. Chạy Kernel MatMul: Đọc $X$ và $W$ từ RAM $\rightarrow$ Tính $t_1 = XW$ $\rightarrow$ Ghi $t_1$ về RAM.
2. Chạy Kernel BiasAdd: Đọc $t_1$ và $b$ từ RAM $\rightarrow$ Tính $t_2 = t_1 + b$ $\rightarrow$ Ghi $t_2$ về RAM.
3. Chạy Kernel GELU: Đọc $t_2$ từ RAM $\rightarrow$ Tính $Y = \text{GELU}(t_2)$ $\rightarrow$ Ghi $Y$ về RAM.
* *Tổng chi phí:* 3 lần đọc RAM, 3 lần ghi RAM.

#### Quy trình khi CÓ Fusion (Linear + Bias + GELU Fusion):
1. Chạy Fused Kernel duy nhất:
   * Đọc $X$, $W$, và $b$ từ RAM.
   * Tính toán từng phần tử tích lũy $t_{1,ij}$ trên thanh ghi CPU.
   * Cộng trực tiếp $b_j$ ngay trên thanh ghi: $t_{2,ij} = t_{1,ij} + b_j$.
   * Áp dụng ngay hàm kích hoạt GELU trên thanh ghi bằng vector hóa SIMD: $Y_{ij} = \text{GELU}(t_{2,ij})$.
   * Ghi kết quả cuối cùng $Y$ về RAM.
* *Tổng chi phí:* 1 lần đọc RAM, 1 lần ghi RAM. Tiết kiệm tối đa chu kỳ chờ của CPU.

---

## 3. Xác định Tile Size tối ưu trên i5-1135G7

Đối với CPU Intel Core i5-1135G7 có L2 Cache là 1.25MB (1280 KB) cho mỗi nhân vật lý:
* Nếu dùng kiểu dữ liệu FP32 (4 bytes/phần tử): Ta thiết lập kích thước tile sao cho tổng dữ liệu khối $A$ và $B$ chiếm khoảng 70% dung lượng L2 (để dành 30% cho dữ liệu phụ và kết quả $C$):
  $$(B_m \times B_k + B_k \times B_n) \times 4 \le 900 \times 1024 \text{ bytes}$$
* Nếu chọn $B_k = 64$ (phù hợp với Vector Registers):
  $$(B_m + B_n) \times 256 \le 921600 \implies B_m + B_n \le 3600$$
  Chúng ta có thể chọn $B_m = 128, B_n = 2048$ làm cấu hình tối ưu để tối đa hóa Spatial Locality cho ma trận $B$.
