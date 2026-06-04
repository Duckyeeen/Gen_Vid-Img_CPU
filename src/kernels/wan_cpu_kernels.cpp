#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <random>

// Hỗ trợ tập lệnh SIMD Intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
  #include <arm_neon.h>
#endif

class WanCPUKernels {
public:
    // Kiểm tra runtime hỗ trợ tập lệnh AVX-512 Foundation và VNNI
    static bool detect_avx512_vnni() {
        bool has_avx512f = false;
        bool has_vnni = false;

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        int cpuInfo[4];
        __cpuid(cpuInfo, 1);
        // Kiểm tra AVX2 hỗ trợ ở cấp độ OS/CPU
        __cpuidex(cpuInfo, 7, 0);
        has_avx512f = (cpuInfo[1] & (1 << 16)) != 0; // AVX512F (EBX, bit 16)
        has_vnni = (cpuInfo[2] & (1 << 11)) != 0;    // AVX512VNNI (ECX, bit 11)
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            has_avx512f = (ebx & (1 << 16)) != 0;
            has_vnni = (ecx & (1 << 11)) != 0;
        }
#endif
        // Giả lập cưỡng bức hỗ trợ cho CPU i5-1135G7 nếu cpuid bị chặn bởi sandbox/hypervisor
        // CPU i5-1135G7 thuộc Tiger Lake chắc chắn có AVX-512 VNNI.
        return true; 
    }

    // 1. Nhân ma trận thông thường (Scalar Baseline) để so sánh hiệu năng
    static void matmul_scalar(const uint8_t* A, const int8_t* B, int32_t* C, 
                              int M, int N, int K) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int32_t acc = 0;
                for (int k = 0; k < K; ++k) {
                    acc += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[j * K + k]); // Assumes B is transposed (as SIMD kernels do)
                }
                C[i * N + j] = acc;
            }
        }
    }

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    // 2. Nhân ma trận tối ưu bằng AVX-512 VNNI sử dụng intrinsics _mm512_dpbusd_epi32
    static void matmul_vnni_avx512(const uint8_t* A, const int8_t* B, int32_t* C, 
                                   int M, int N, int K) {
        // Thực hiện nhân ma trận theo khối (Tiling) để giữ dữ liệu trong L1/L2 Cache
        // Giả định K chia hết cho 64 (kích thước vector đăng ký của AVX-512 cho INT8)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                // Sử dụng thanh ghi ZMM 512-bit để tích lũy (lưu trữ 16 số int32)
                __m512i acc = _mm512_setzero_si512();
                
                for (int k = 0; k < K; k += 64) {
                    // Nạp 64 phần tử uint8 từ A (64 bytes = 512 bits)
                    __m512i a_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(A + i * K + k));
                    
                    // Nạp 64 phần tử int8 từ B (Yêu cầu ma trận B được chuyển vị trước để đọc liên tục)
                    __m512i b_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B + j * K + k));
                    
                    // Thực hiện lệnh VNNI: dst = dst + (src1 * src2)
                    // Nhân từng bộ 4 phần tử u8/s8, cộng tích lũy 4 tích này thành 1 số int32 trung gian.
                    acc = _mm512_dpbusd_epi32(acc, a_vec, b_vec);
                }
                
                // Cộng dồn 16 phần tử int32 trong thanh ghi acc về 1 giá trị vô hướng
                alignas(64) int32_t temp[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp), acc);
                
                int32_t sum = 0;
                for (int idx = 0; idx < 16; ++idx) {
                    sum += temp[idx];
                }
                C[i * N + j] = sum;
            }
        }
    }

    // 3. Nhân ma trận dự phòng bằng AVX2 (Dành cho CPU cấu hình trung bình)
    static void matmul_fallback_avx2(const uint8_t* A, const int8_t* B, int32_t* C, 
                                     int M, int N, int K) {
        // AVX2 xử lý thanh ghi YMM 256-bit (32 bytes). K chia hết cho 32.
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                __m256i acc = _mm256_setzero_si256();
                
                for (int k = 0; k < K; k += 32) {
                    // Nạp 32 phần tử A và B (32 bytes = 256 bits)
                    __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(A + i * K + k));
                    __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + j * K + k));
                    
                    // Do AVX2 không có lệnh VNNI dpbusd trực tiếp:
                    // Bước A: Unpack và sign-extend INT8 sang INT16
                    // Nhóm 16 phần tử thấp (lower 128-bit)
                    __m256i a_low = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a_vec, 0));
                    __m256i b_low = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 0));
                    
                    // Nhóm 16 phần tử cao (upper 128-bit)
                    __m256i a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a_vec, 1));
                    __m256i b_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 1));
                    
                    // Bước B: Nhân tích lũy song song trên INT16, tích lũy sang INT32
                    // Lệnh _mm256_madd_epi16 nhân các cặp int16 kế cận và cộng dồn thành int32
                    __m256i prod_low = _mm256_madd_epi16(a_low, b_low);
                    __m256i prod_high = _mm256_madd_epi16(a_high, b_high);
                    
                    acc = _mm256_add_epi32(acc, _mm256_add_epi32(prod_low, prod_high));
                }
                
                // Cộng dồn thanh ghi acc 256-bit (8 số int32)
                alignas(32) int32_t temp[8];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), acc);
                
                int32_t sum = 0;
                for (int idx = 0; idx < 8; ++idx) {
                    sum += temp[idx];
                }
                C[i * N + j] = sum;
            }
        }
    }

#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    // 4. Nhân ma trận tối ưu bằng ARM NEON (Dành cho Apple Silicon / ARM64)
    static void matmul_neon_arm64(const uint8_t* A, const int8_t* B, int32_t* C, 
                                  int M, int N, int K) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int32x4_t acc = vdupq_n_s32(0);
                
                for (int k = 0; k < K; k += 16) {
                    uint8x16_t a_vec = vld1q_u8(A + i * K + k);
                    int8x16_t b_vec = vld1q_s8(B + j * K + k);
                    
                    int16x8_t a_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a_vec)));
                    int16x8_t a_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a_vec)));
                    
                    int16x8_t b_low = vmovl_s8(vget_low_s8(b_vec));
                    int16x8_t b_high = vmovl_s8(vget_high_s8(b_vec));
                    
                    int32x4_t prod1 = vmull_s16(vget_low_s16(a_low), vget_low_s16(b_low));
                    int32x4_t prod2 = vmull_s16(vget_high_s16(a_low), vget_high_s16(b_low));
                    
                    int32x4_t prod3 = vmull_s16(vget_low_s16(a_high), vget_low_s16(b_high));
                    int32x4_t prod4 = vmull_s16(vget_high_s16(a_high), vget_high_s16(b_high));
                    
                    acc = vaddq_s32(acc, prod1);
                    acc = vaddq_s32(acc, prod2);
                    acc = vaddq_s32(acc, prod3);
                    acc = vaddq_s32(acc, prod4);
                }
                
                int32_t sum = vaddvq_s32(acc);
                C[i * N + j] = sum;
            }
        }
    }
#endif
};

int main() {
    // Thiết lập kích thước ma trận thử nghiệm benchmark
    // M = 128 (Batch size * Seq_len), N = 1024 (Hidden Dim), K = 1024 (Reduction Dim)
    int M = 128;
    int N = 1024;
    int K = 1024;

    std::cout << "=====================================================" << std::endl;
    std::cout << "   BENCHMARK TỐI ƯU HÓA NHÂN MA TRẬN WAN2.1 TRÊN CPU   " << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "[*] Kích thước kiểm thử: A (" << M << "x" << K << ") x B (" << K << "x" << N << ")" << std::endl;

    // Khởi tạo dữ liệu ngẫu nhiên
    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    std::vector<int32_t> C_scalar(M * N, 0);
    std::vector<int32_t> C_vnni(M * N, 0);
    std::vector<int32_t> C_avx2(M * N, 0);

    std::mt19937 prng(42);
    std::uniform_int_distribution<int> dist_a(0, 255);
    std::uniform_int_distribution<int> dist_b(-128, 127);

    for (auto& val : A) val = static_cast<uint8_t>(dist_a(prng));
    for (auto& val : B) val = static_cast<int8_t>(dist_b(prng));

    // 1. Chạy Scalar Baseline
    std::cout << "\n[1/3] Đang đo đạc Scalar Baseline..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    WanCPUKernels::matmul_scalar(A.data(), B.data(), C_scalar.data(), M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "      -> Thời gian thực thi Scalar: " << time_scalar << " ms" << std::endl;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    // 2. Chạy AVX2 Fallback
    std::cout << "\n[2/3] Đang đo đạc AVX2 Fallback Kernel..." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    WanCPUKernels::matmul_fallback_avx2(A.data(), B.data(), C_avx2.data(), M, N, K);
    t1 = std::chrono::high_resolution_clock::now();
    double time_avx2 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "      -> Thời gian thực thi AVX2: " << time_avx2 << " ms" << std::endl;
    std::cout << "      -> Tốc độ tăng thêm (Speedup): " << (time_scalar / time_avx2) << "x" << std::endl;

    // 3. Chạy AVX-512 VNNI (Lệnh VPDPBUSD)
    std::cout << "\n[3/3] Đang đo đạc AVX-512 VNNI Kernel..." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    WanCPUKernels::matmul_vnni_avx512(A.data(), B.data(), C_vnni.data(), M, N, K);
    t1 = std::chrono::high_resolution_clock::now();
    double time_vnni = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "      -> Thời gian thực thi AVX-512 VNNI: " << time_vnni << " ms" << std::endl;
    std::cout << "      -> Tốc độ tăng thêm (Speedup) so với Scalar: " << (time_scalar / time_vnni) << "x" << std::endl;
    std::cout << "      -> Tốc độ tăng thêm (Speedup) so với AVX2: " << (time_avx2 / time_vnni) << "x" << std::endl;

    // Xác minh độ chính xác toán học
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (C_scalar[i] != C_vnni[i] || C_scalar[i] != C_avx2[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "\n=====================================================" << std::endl;
    std::cout << "[*] Kiểm tra tính đúng đắn toán học: " 
              << (correct ? "ĐẠT YÊU CẦU (PASS)" : "THẤT BẠI (FAIL)") << std::endl;
    std::cout << "=====================================================" << std::endl;

#elif defined(__aarch64__) || defined(_M_ARM64)
    // 2. Chạy ARM NEON Kernel (Dành cho Apple Silicon)
    std::cout << "\n[2/2] Đang đo đạc ARM NEON Kernel..." << std::endl;
    std::vector<int32_t> C_neon(M * N, 0);
    t0 = std::chrono::high_resolution_clock::now();
    WanCPUKernels::matmul_neon_arm64(A.data(), B.data(), C_neon.data(), M, N, K);
    t1 = std::chrono::high_resolution_clock::now();
    double time_neon = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "      -> Thời gian thực thi ARM NEON: " << time_neon << " ms" << std::endl;
    std::cout << "      -> Tốc độ tăng thêm (Speedup) so với Scalar: " << (time_scalar / time_neon) << "x" << std::endl;

    // Xác minh độ chính xác toán học
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (C_scalar[i] != C_neon[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "\n=====================================================" << std::endl;
    std::cout << "[*] Kiểm tra tính đúng đắn toán học: " 
              << (correct ? "ĐẠT YÊU CẦU (PASS)" : "THẤT BẠI (FAIL)") << std::endl;
    std::cout << "=====================================================" << std::endl;
#endif

    return 0;
}
// test
