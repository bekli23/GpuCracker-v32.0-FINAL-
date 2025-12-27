# --- CONFIGURARE COMPILATOR ---
NVCC = nvcc
CXX = g++

# --- ARHITECTURI (UNIVERSAL) ---
# Include de la GTX 900 (sm_50) pana la H100 (sm_90)
ARCH_FLAGS = -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_61,code=sm_61 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90
             # Decomenteaza cand apare CUDA 12.8+ pe Linux:
             # -gencode arch=compute_100,code=sm_100

# --- FLAGURI ---
CFLAGS = -O3 -std=c++17 -Xcompiler -fopenmp -w
LDFLAGS = -lOpenCL -lcrypto -lssl -lsecp256k1 -lcurand

# --- SURSE ---
SOURCES = main.cu kernel_ops.cu GpuCore.cu cuda_provider.cu
OBJECTS = $(SOURCES:.cu=.o)
TARGET = GpuCracker

# --- REGULI ---
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(ARCH_FLAGS) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(ARCH_FLAGS) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean