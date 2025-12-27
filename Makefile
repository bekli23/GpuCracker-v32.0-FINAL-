# ============================================================================
# GpuCracker v32.0 (FINAL) - Makefile for Linux (Ubuntu 22.04)
# ============================================================================

# --- CONFIGURARE COMPILATOARE ---
NVCC = nvcc
CXX  = g++

# --- ARHITECTURI GPU (UNIVERSAL COMPATIBILITY) ---
# Include suport complet de la Maxwell (sm_50) până la Hopper (sm_90)
ARCH_FLAGS = -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_61,code=sm_61 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_87,code=sm_87 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90

# --- FLAGURI DE COMPILARE ---
# -O3: Optimizare maximă pentru viteză
# -std=c++17: Standardul de limbaj necesar
# -Xcompiler -fopenmp: Activează suportul multi-core CPU (OpenMP)
# -w: Ignoră warning-urile pentru o compilare mai curată
CFLAGS   = -O3 -std=c++17 -Xcompiler -fopenmp -w
LDFLAGS  = -lOpenCL -lcrypto -lssl -lsecp256k1 -lcurand

# --- SURSE ȘI OBIECTE ---
# Include mnemonic_gpu.cu pentru suportul BIP39 complet pe GPU
SOURCES  = main.cu kernel_ops.cu GpuCore.cu cuda_provider.cu mnemonic_gpu.cu
OBJECTS  = $(SOURCES:.cu=.o)

# --- TARGETURI FINALE ---
TARGET      = GpuCracker
BUILD_BLOOM = build_bloom

# --- REGULI DE CONSTRUCȚIE ---

all: $(TARGET) $(BUILD_BLOOM)

# Compilarea executabilului principal GpuCracker
$(TARGET): $(OBJECTS)
	$(NVCC) $(ARCH_FLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compilarea utilitarului pentru filtre Bloom
$(BUILD_BLOOM): build_bloom.cpp
	$(CXX) -O3 build_bloom.cpp -o $@ -lssl -lcrypto

# Regula pentru transformarea fișierelor .cu în obiecte .o
%.o: %.cu
	$(NVCC) $(ARCH_FLAGS) $(CFLAGS) -c $< -o $@

# Curățarea fișierelor temporare și a executabilelor
clean:
	rm -f *.o $(TARGET) $(BUILD_BLOOM)

.PHONY: all clean