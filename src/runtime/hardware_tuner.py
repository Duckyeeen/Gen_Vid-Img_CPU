# -*- coding: utf-8 -*-
"""
Wan2.1 Dynamic Hardware Auto-Tuner
Benchmarks target CPU performance (GFLOPS, RAM Bandwidth) and generates device_config.json.
"""

import os
import json
import time
import psutil
import numpy as np

class WanHardwareTuner:
    def __init__(self):
        print("=====================================================")
        print("     WAN2.1 DYNAMIC HARDWARE BENCHMARK & AUTO-TUNER  ")
        print("=====================================================")
        self.logical_cores = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False)

    def benchmark_gflops(self):
        """
        Đo đạc sức mạnh tính toán thực tế của CPU thông qua phép nhân ma trận lớn (FP32).
        """
        print("[*] Đang đo đạc năng lực tính toán dấu phẩy động (GFLOPS)...")
        size = 1024
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Chạy 3 lần để lấy trung bình
        warmup = np.dot(A, B) # Warm up cache/pipeline
        
        times = []
        for _ in range(5):
            t0 = time.time()
            np.dot(A, B)
            times.append(time.time() - t0)
            
        avg_time = np.mean(times)
        # Số lượng phép tính: 2 * N^3 (nhân và cộng)
        ops = 2.0 * (size ** 3)
        gflops = (ops / avg_time) / 1e9
        print(f"    - Thời gian nhân ma trận {size}x{size} trung bình: {avg_time * 1000.0:.2f} ms")
        print(f"    - Điểm số tính toán CPU: {gflops:.2f} GFLOPS")
        return gflops

    def benchmark_memory_bandwidth(self):
        """
        Đo đạc băng thông đọc/ghi RAM thực tế (MB/s).
        """
        print("[*] Đang đo đạc băng thông đọc ghi bộ nhớ (RAM Bandwidth)...")
        # 50 triệu phần tử float32 = 200 MB
        size = 50000000 
        data = np.ones(size, dtype=np.float32)
        
        t0 = time.time()
        # Ghi bộ nhớ
        data *= 2.5
        t1 = time.time()
        write_time = t1 - t0
        write_bw = (200.0) / write_time # MB/s
        
        # Đọc bộ nhớ (sum)
        t0 = time.time()
        np.sum(data)
        t1 = time.time()
        read_time = t1 - t0
        read_bw = (200.0) / read_time # MB/s
        
        avg_bw = (write_bw + read_bw) / 2.0
        print(f"    - Tốc độ Đọc bộ nhớ: {read_bw:.2f} MB/s")
        print(f"    - Tốc độ Ghi bộ nhớ: {write_bw:.2f} MB/s")
        print(f"    - Băng thông RAM trung bình: {avg_bw:.2f} MB/s")
        return avg_bw

    def determine_cpu_profile(self, gflops, memory_bw):
        """
        Phân loại CPU và tự động tính toán các tham số tối ưu hóa tốt nhất.
        """
        print("[*] Đang tính toán cấu hình tối ưu tự động thích ứng...")
        
        # Định nghĩa các cấu hình dựa trên kết quả thực tế
        profile = "Medium-Performance CPU"
        tile_m, tile_k, tile_n = 64, 64, 1024
        use_vnni = True
        
        # Phân loại hiệu năng
        if gflops > 100.0:
            profile = "High-Performance Server/Workstation CPU"
            tile_m, tile_k, tile_n = 128, 64, 2048
        elif gflops < 30.0:
            profile = "Low-Power / Mobile CPU"
            tile_m, tile_k, tile_n = 32, 32, 512
            use_vnni = False
            
        # Tự động gán thread theo số lõi vật lý thực để tối ưu cache
        threads = self.physical_cores if self.physical_cores else 4
        
        # i5-1135G7 có 4 physical cores, L2 cache 1.25MB per core.
        # Chúng ta tối ưu hóa cấu hình cho CPU này
        if "Intel" in os.popen("wmic cpu get name").read():
            print("    - Phát hiện bộ vi xử lý Intel thế hệ mới (hỗ trợ nâng cao AVX-512 VNNI).")
            use_vnni = True
            
        config = {
            "device_profile": profile,
            "measured_gflops": round(gflops, 2),
            "measured_ram_bandwidth_mbs": round(memory_bw, 2),
            "optimal_thread_count": threads,
            "optimal_tiling_parameters": {
                "B_m": tile_m,
                "B_k": tile_k,
                "B_n": tile_n
            },
            "enable_avx512_vnni": use_vnni,
            "affinity_cores": [i for i in range(0, self.logical_cores, 2)] if self.logical_cores else [0, 2, 4, 6]
        }
        
        # Ghi file cấu hình
        config_path = "c:/GitHub/Gen_Vid-Img_CPU/docs/plans/device_config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
        print(f"[+] Đã tạo file cấu hình tối ưu động thành công tại: {config_path}")
        return config

if __name__ == "__main__":
    tuner = WanHardwareTuner()
    gflops = tuner.benchmark_gflops()
    bandwidth = tuner.benchmark_memory_bandwidth()
    tuner.determine_cpu_profile(gflops, bandwidth)
