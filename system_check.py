import torch
import sys
import platform
import os

print("🔹 SYSTEM INFORMATION 🔹")
print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# CPU Info
print("\n🔹 CPU INFORMATION 🔹")
print(f"Processor: {platform.processor()}")

# Check CUDA availability
print("\n🔹 CUDA & GPU INFORMATION 🔹")
print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version (from PyTorch): {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")

    for i in range(torch.cuda.device_count()):
        print(f"\n🚀 GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"   CUDA Device ID: {i}")
        print(f"   Total Memory: {round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2)} GB")
        print(f"   Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
        print(f"   Max Threads per Multiprocessor: {torch.cuda.get_device_properties(i).max_threads_per_multi_processor}")

    print(f"\nCurrent CUDA Device: {torch.cuda.current_device()} (Device ID)")
    print(f"Allocated GPU Memory: {round(torch.cuda.memory_allocated() / (1024 ** 3), 2)} GB")
    print(f"Cached GPU Memory: {round(torch.cuda.memory_reserved() / (1024 ** 3), 2)} GB")
else:
    print("⚠️ CUDA is not available! Check installation.")

print("\n✅ System check completed!")
