#!/bin/bash
echo "====================================================="
echo "   COMPILING WAN2.1 KERNELS FOR APPLE SILICON (M1)   "
echo "====================================================="
clang++ -O3 -mcpu=apple-m1 -std=c++14 src/kernels/wan_cpu_kernels.cpp -o wan_kernels_mac
if [ $? -eq 0 ]; then
    echo "[+] Compilation successful!"
    echo "[*] Running Benchmark..."
    echo "-----------------------------------------------------"
    ./wan_kernels_mac
else
    echo "[-] Compilation failed!"
    exit 1
fi
