# -*- coding: utf-8 -*-
"""
Wan2.1 (wan2gp) Custom Graph Optimizer & SmoothQuant Simulator
Mathematically models the activation-weight quantization and graph fusion passes.
"""

import os
import json
import numpy as np

class WanGraphOptimizer:
    """
    Trình tối ưu hóa đồ thị ANF và giả lập lượng tử hóa toán học cho Wan2.1.
    """
    def __init__(self, model_name="wan2gp"):
        self.model_name = model_name
        self.nodes = {}
        self.edges = []
        self.passes_applied = []

    def load_graph_from_mindspore(self):
        """
        Khởi tạo đồ thị mô phỏng các block DiT / 3D-VAE của Wan2.1.
        """
        self.nodes = {
            "input_act": {"type": "Parameter", "name": "Activation_Input (X)", "shape": [128, 1024]},
            "weight_linear": {"type": "Parameter", "name": "Linear_Weight (W)", "shape": [1024, 1024]},
            "matmul_op": {"type": "CNode", "name": "MatMul", "inputs": ["input_act", "weight_linear"]},
            "bias_op": {"type": "CNode", "name": "BiasAdd", "inputs": ["matmul_op"]},
            "transpose_op": {"type": "CNode", "name": "Transpose", "inputs": ["bias_op"]},
            "softmax_op": {"type": "CNode", "name": "Softmax", "inputs": ["transpose_op"]}
        }
        self.rebuild_edges()
        print(f"[*] Đã tải đồ thị {self.model_name} thành công.")

    def rebuild_edges(self):
        self.edges = []
        for node_id, info in self.nodes.items():
            if "inputs" in info:
                for inp in info["inputs"]:
                    self.edges.append((inp, node_id))

    def run_smoothquant_simulation(self, alpha=0.5):
        """
        Giả lập toán học SmoothQuant: Y = X * W = (X * S^-1) * (S * W)
        Kênh nào của X có outliers lớn sẽ được kéo xuống, Weight W sẽ hấp thụ phần scale đó.
        """
        print(f"\n[+] Đang giả lập SmoothQuant Pass với alpha = {alpha}...")
        
        # Khởi tạo dữ liệu ngẫu nhiên mô phỏng có chứa Outliers lớn ở một vài kênh
        np.random.seed(42)
        X = np.random.randn(128, 1024)
        # Tạo outliers nhân tạo ở kênh 10 và 45 (giá trị vọt lên 95.0 và 120.0)
        X[:, 10] *= 50.0
        X[:, 45] *= 60.0
        
        W = np.random.randn(1024, 1024)
        
        # Tính toán ma trận làm mượt đường chéo S_j
        max_x = np.max(np.abs(X), axis=0) # Kích thước 1024
        max_w = np.max(np.abs(W), axis=0) # Kích thước 1024
        
        # Công thức: S_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
        S = np.power(max_x, alpha) / np.power(max_w, 1.0 - alpha)
        # Tránh chia cho 0
        S[S == 0] = 1e-5
        
        # Làm mượt Activation và Weight
        X_smooth = X / S
        W_smooth = W * S[:, np.newaxis]
        
        # Đo đạc lỗi lượng tử hóa trước và sau khi làm mượt
        # A. Trước khi làm mượt (Lượng tử hóa trực tiếp)
        scale_x_orig = np.max(np.abs(X)) / 127.0
        q_x_orig = np.round(X / scale_x_orig)
        dq_x_orig = q_x_orig * scale_x_orig
        orig_error = np.mean((X - dq_x_orig) ** 2)
        
        # B. Sau khi làm mượt SmoothQuant
        scale_x_smooth = np.max(np.abs(X_smooth)) / 127.0
        q_x_smooth = np.round(X_smooth / scale_x_smooth)
        dq_x_smooth = q_x_smooth * scale_x_smooth
        smooth_error = np.mean((X_smooth - dq_x_smooth) ** 2)
        
        print(f"    - Dải động cực đại của X trước SmoothQuant: {np.max(np.abs(X)):.4f}")
        print(f"    - Dải động cực đại của X sau SmoothQuant: {np.max(np.abs(X_smooth)):.4f}")
        print(f"    - Sai số lượng tử hóa bình phương (MSE) TRƯỚC SmoothQuant: {orig_error:.6f}")
        print(f"    - Sai số lượng tử hóa bình phương (MSE) SAU SmoothQuant: {smooth_error:.6f}")
        reduction = (orig_error - smooth_error) / orig_error * 100.0
        print(f"    - Tỷ lệ giảm sai số (Quantization Error Reduction): {reduction:.2f}%")
        
        # Cập nhật thông tin đồ thị
        self.nodes["smooth_quant_op"] = {
            "type": "CNode",
            "name": "SmoothQuant_Scale_Fusion",
            "inputs": ["input_act", "weight_linear"],
            "smooth_alpha": alpha,
            "quantization_error_reduction": f"{reduction:.2f}%"
        }
        self.passes_applied.append("SmoothQuantPass")

    def run_operator_fusion_pass(self):
        """
        Hợp nhất: MatMul + BiasAdd + Transpose -> Fused_MatMul_Bias_Transpose (VNNI)
        """
        print("\n[+] Đang chạy Operator Fusion Pass...")
        fused_node_id = "fused_matmul_bias_transpose"
        
        self.nodes[fused_node_id] = {
            "type": "CNode",
            "name": "Fused_MatMul_Bias_Transpose (VNNI)",
            "inputs": ["input_act", "weight_linear"],
            "fused_operations": ["MatMul", "BiasAdd", "Transpose"]
        }
        
        # Cập nhật đầu vào cho Softmax
        if "softmax_op" in self.nodes:
            self.nodes["softmax_op"]["inputs"] = [fused_node_id]
            
        # Xóa các nút đã gộp
        for op in ["matmul_op", "bias_op", "transpose_op"]:
            if op in self.nodes:
                del self.nodes[op]
                
        self.rebuild_edges()
        self.passes_applied.append("OperatorFusionPass")
        print("[*] Đã tối ưu hóa đồ thị. Hợp nhất 3 CNode thành 1 nhân VNNI duy nhất.")

    def export_graph_to_json(self, file_path):
        graph_data = {
            "model": self.model_name,
            "nodes": self.nodes,
            "edges": [{"source": edge[0], "target": edge[1]} for edge in self.edges],
            "passes_applied": self.passes_applied
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4, ensure_ascii=False)
        print(f"[+] Đã xuất đồ thị tối ưu hóa ra file: {file_path}")


if __name__ == "__main__":
    optimizer = WanGraphOptimizer()
    optimizer.load_graph_from_mindspore()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    docs_dir = os.path.join(base_dir, "docs")
    
    # 1. Trực quan đồ thị ban đầu
    optimizer.export_graph_to_json(os.path.join(docs_dir, "graph_before_opt.json"))
    
    # 2. Chạy SmoothQuant mô phỏng toán học
    optimizer.run_smoothquant_simulation(alpha=0.5)
    
    # 3. Chạy fusion đồ thị
    optimizer.run_operator_fusion_pass()
    
    # 4. Trực quan đồ thị sau tối ưu
    optimizer.export_graph_to_json(os.path.join(docs_dir, "graph_after_opt.json"))
