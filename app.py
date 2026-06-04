import gradio as gr
import subprocess
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def run_python_benchmark():
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(["python3", "src/benchmark_runner.py"], cwd=BASE_DIR, env=env, capture_output=True, text=True)
        report_path = os.path.join(BASE_DIR, "docs/plans/performance_report.md")
        report_content = "Không tìm thấy file báo cáo."
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f: report_content = f.read()
        return result.stdout + "\n" + result.stderr, report_content
    except Exception as e: return str(e), "Có lỗi xảy ra"
def run_cpp_kernel():
    try:
        result = subprocess.run(["./compile_mac.sh"], cwd=BASE_DIR, capture_output=True, text=True)
        return result.stdout + "\n" + result.stderr
    except Exception as e: return str(e)
with gr.Blocks(title="Wan2.1 CPU Optimization Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Wan2.1 CPU Optimization Dashboard")
    gr.Markdown("Giao diện điều khiển trực quan cho dự án tối ưu hóa hiệu năng mô hình Wan2.1 trên CPU.")
    with gr.Tab("Hiệu Năng Toàn Cục (Python)"):
        btn_py = gr.Button("⚡ Khởi chạy Benchmark Toàn hệ thống", variant="primary")
        with gr.Row():
            with gr.Column(scale=1): out_py_log = gr.Textbox(label="Logs Hệ thống", lines=20)
            with gr.Column(scale=1): out_py_md = gr.Markdown(label="Báo Cáo Phân Tích")
        btn_py.click(fn=run_python_benchmark, inputs=[], outputs=[out_py_log, out_py_md])
    with gr.Tab("Kiểm Thử Lõi C++ (Phần cứng)"):
        btn_cpp = gr.Button("🔨 Biên dịch & Chạy Kiểm thử C++", variant="primary")
        with gr.Row(): out_cpp_log = gr.Textbox(label="Kết quả đo đạc (C++ Logs)", lines=15)
        btn_cpp.click(fn=run_cpp_kernel, inputs=[], outputs=[out_cpp_log])
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
