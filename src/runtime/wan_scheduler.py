# -*- coding: utf-8 -*-
"""
Wan2.1 Thread Scheduler & Core Affinity Manager
Optimized to dynamically load parameters from docs/plans/device_config.json.
"""

import os
import sys
import json
import time
import psutil

class WanCPUScheduler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.num_logical_cores = psutil.cpu_count(logical=True)
        self.num_physical_cores = psutil.cpu_count(logical=False)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = os.path.join(base_dir, "docs", "plans", "device_config.json")
        
        # Load cấu hình tự động thích ứng
        self.config = self.load_device_config()

    def load_device_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print(f"[+] Đã tải cấu hình phần cứng tối ưu động:")
                print(f"    - CPU Profile: {config.get('device_profile')}")
                print(f"    - Core Affinity đề xuất: {config.get('affinity_cores')}")
                print(f"    - Số luồng OMP tối ưu: {config.get('optimal_thread_count')}")
                return config
            except Exception as e:
                print(f"[-] Lỗi đọc file cấu hình: {str(e)}")
        
        # Cấu hình mặc định dự phòng nếu chưa chạy hardware tuner
        print("[!] Không tìm thấy file cấu hình, sử dụng các tham số dự phòng.")
        return {
            "optimal_thread_count": self.num_physical_cores,
            "affinity_cores": [i for i in range(0, self.num_logical_cores, 2)] if self.num_logical_cores else [0, 2, 4, 6]
        }

    def pin_to_physical_cores(self):
        """
        Khóa các luồng tính toán vào các lõi CPU vật lý dựa trên cấu hình tự động dò tìm.
        """
        print("[+] Đang thiết lập Core Affinity cho tiến trình...")
        if not hasattr(self.process, 'cpu_affinity'):
            print("[-] Hệ điều hành này không hỗ trợ thiết lập Core Affinity qua psutil (ví dụ macOS). Bỏ qua tính năng này.")
            return False
        try:
            core_ids = self.config.get("affinity_cores", [0, 2, 4, 6])
            
            # Đảm bảo các core đề xuất không vượt quá số core thực tế của máy chạy
            valid_core_ids = [c for c in core_ids if c < self.num_logical_cores]
            
            self.process.cpu_affinity(valid_core_ids)
            applied_affinity = self.process.cpu_affinity()
            print(f"[+] Đã khóa tiến trình thành công vào các lõi CPU: {applied_affinity}")
            print(f"[!] Đã kích hoạt Warm Cache mode (Ngăn chặn Windows đổi core ngẫu nhiên).")
            return True
        except Exception as e:
            print(f"[-] Không thể thiết lập Core Affinity: {str(e)}")
            return False

    def optimize_openmp_threads(self):
        """
        Cấu hình số luồng chạy OpenMP/MKL từ file cấu hình tối ưu.
        """
        num_threads = str(self.config.get("optimal_thread_count", self.num_physical_cores))
        os.environ["OMP_NUM_THREADS"] = num_threads
        os.environ["MKL_NUM_THREADS"] = num_threads
        os.environ["OPENBLAS_NUM_THREADS"] = num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = num_threads
        print(f"[+] Đã cấu hình môi trường OMP_NUM_THREADS = {num_threads}.")

    def monitor_cpu_usage(self, duration_sec=3, interval=1.0):
        print(f"\n[*] Đang giám sát sử dụng CPU trong {duration_sec} giây...")
        start_time = time.time()
        step = 1
        
        has_affinity = hasattr(self.process, 'cpu_affinity')
        try:
            current_affinity = self.process.cpu_affinity() if has_affinity else []
        except Exception:
            current_affinity = []
            
        while time.time() - start_time < duration_sec:
            cpu_percentages = psutil.cpu_percent(interval=interval, percpu=True)
            print(f"    [Đo đạc lần {step}] Sử dụng CPU trên từng core (%):")
            for idx, percent in enumerate(cpu_percentages):
                bar = "#" * int(percent / 5)
                if has_affinity:
                    tag = "(Real Core - PINNED)" if idx in current_affinity else "(HyperThread - IDLE)"
                else:
                    tag = "(Core)"
                print(f"      - Core {idx} {tag:22}: [{percent:5.1f}%] {bar}")
            step += 1
            print("-" * 50)


if __name__ == "__main__":
    scheduler = WanCPUScheduler()
    scheduler.pin_to_physical_cores()
    scheduler.optimize_openmp_threads()
    scheduler.monitor_cpu_usage(duration_sec=3, interval=1.0)
