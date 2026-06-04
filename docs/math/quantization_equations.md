# Các Phương trình Toán học Lượng tử hóa & SmoothQuant

Tài liệu này lưu trữ các phương trình toán học nền tảng được áp dụng để nén và tối ưu hóa các tensor của mô hình **Wan2.1** từ FP32 sang INT8 trên CPU.

---

## 1. Lượng tử hóa Tuyến tính (Linear Quantization)

Ánh xạ một giá trị số thực $f \in [\alpha, \beta]$ sang một giá trị số nguyên $q \in [q_{min}, q_{max}]$ (với INT8, $[q_{min}, q_{max}] = [-128, 127]$):

### Công thức Lượng tử hóa (Quantization):
$$q = \text{clamp}\left( \text{round}\left( \frac{f}{\text{scale}} \right) + \text{zero\_point}, \ q_{min}, \ q_{max} \right)$$

### Công thức Giải lượng tử hóa (Dequantization):
$$\hat{f} = \text{scale} \cdot (q - \text{zero\_point})$$

### Tính toán các Tham số Lượng tử hóa (Scale & Zero-Point):
$$\text{scale} = \frac{\beta - \alpha}{q_{max} - q_{min}}$$
$$\text{zero\_point} = \text{round}\left( \frac{-\alpha}{\text{scale}} \right) + q_{min}$$

Trong đó:
* $\text{clamp}(x, a, b) = \max(a, \min(x, b))$
* Đối với lượng tử hóa đối xứng (Symmetric Quantization), $\text{zero\_point} = 0$ và $\text{scale} = \frac{\max(|\alpha|, |\beta|)}{q_{max}}$.

---

## 2. SmoothQuant: Làm mượt Outliers

Đối với phép nhân ma trận trong khối Attention $Y = XW$, Activation $X$ thường xuất hiện các giá trị cực đại (outliers) ở một vài kênh (channels) cố định, trong khi Weight $W$ phân bố đều.

### Nguyên lý Giao hoán với Ma trận Đường chéo $S$:
Chúng ta chèn ma trận đường chéo làm mượt $S = \text{diag}(s_1, s_2, \dots, s_C)$ vào giữa phép nhân:
$$Y = XW = (X S^{-1}) (S W) = \hat{X} \hat{W}$$

Trong đó:
* $\hat{X} = X S^{-1}$ (Làm mượt activation bằng cách chia nhỏ các outliers)
* $\hat{W} = S W$ (Hấp thụ hệ số scale vào weights)

### Công thức Tính toán Hệ số Làm mượt $s_j$ cho kênh $j$:
$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

Với $\alpha \in [0, 1]$ là tham số điều khiển mức độ chuyển dịch độ khó lượng tử hóa:
* $\alpha = 1$: Chuyển toàn bộ độ khó sang Weight (Activation cực kỳ mượt, dễ lượng tử hóa).
* $\alpha = 0$: Chuyển toàn bộ độ khó sang Activation.
* $\alpha = 0.5$: Điểm tối ưu thực nghiệm, chia đều độ khó lượng tử hóa cho cả hai ma trận.

---

## 3. Phân tích Sai số Lượng tử hóa (Quantization Error)

Sai số lượng tử hóa của một phần tử đơn lẻ $\epsilon = f - \hat{f}$ bị giới hạn bởi:
$$|\epsilon| \le \frac{\text{scale}}{2}$$

Sai số bình phương trung bình kỳ vọng (Mean Squared Quantization Error - MSQE) trên một phân phối đồng đều được tính bằng công thức lý thuyết thông tin:
$$\text{MSQE} = E[\epsilon^2] = \frac{\text{scale}^2}{12}$$

Do đó, mục tiêu của các thuật toán tối ưu (như SmoothQuant) là tìm cách thu hẹp dải động $[\alpha, \beta]$ (giảm $\text{scale}$) của các giá trị thực để giảm thiểu tối đa $\text{MSQE}$.
