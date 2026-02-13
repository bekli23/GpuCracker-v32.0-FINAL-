# ============================================================================
# GpuCracker v40.0 - Makefile (Linux - Robust)
# Fix: Eliminare duplicate obiecte (sort) și multiple main
# ============================================================================

# --- CĂI CUDA ---
CUDA_PATH ?= /usr/local/cuda

# --- COMPILATOARE ---
CXX  := g++
NVCC := $(CUDA_PATH)/bin/nvcc

# --- FLAGURI ---
INCLUDES = -I. -I$(CUDA_PATH)/include
# -fpermissive ajută la compatibilitatea cu cod C++ mai vechi/permisiv
CXXFLAGS = -O3 -std=c++17 -fopenmp -pthread -w -fpermissive $(INCLUDES)

# Arhitecturi GPU: sm_86 (RTX 30xx), sm_89 (RTX 40xx)
NVCCFLAGS = -O3 -std=c++17 -Xcompiler -fopenmp -w $(INCLUDES) \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89

# --- LIBRĂRII ---
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lOpenCL -lcrypto -lssl -lsecp256k1 -lcurand -pthread -fopenmp

# --- DEFINIRE ȚINTE (TARGETS) ---
TARGET_APP      = GpuCracker
TARGET_BLOOM    = build_bloom
TARGET_RECOVER  = recover_tool
TARGET_SEED2PRIV= seed2priv_tool

# --- SURSE AUTOMATE ---
SRCS_CPP = $(wildcard *.cpp)
SRCS_CU  = $(wildcard *.cu)

# --- FILTRARE SURSE PENTRU GPUCRACKER ---
# Excludem fișierele care au propriul main()
EXCLUDE_LIST = build_bloom.cpp recover.cpp seed2priv_main.cpp
SRCS_APP_CPP = $(filter-out $(EXCLUDE_LIST), $(SRCS_CPP))

# Obiecte pentru GpuCracker
# FIX: Folosim $(sort ...) pentru a elimina duplicatele (ex: main.o)
OBJS_APP = $(sort $(SRCS_APP_CPP:.cpp=.o) $(SRCS_CU:.cu=.o))

# --- REGULI ---

all: $(TARGET_APP) $(TARGET_BLOOM) $(TARGET_RECOVER) $(TARGET_SEED2PRIV)

# 1. GpuCracker (Main App)
$(TARGET_APP): $(OBJS_APP)
	@echo "[LINK] $@"
	$(CXX) $(OBJS_APP) -o $@ $(LDFLAGS)

# 2. Build Bloom Tool
$(TARGET_BLOOM): build_bloom.o
	@echo "[LINK] $@"
	$(CXX) build_bloom.o -o $@ $(LDFLAGS)

# 3. Recover Tool
$(TARGET_RECOVER): recover.o
	@echo "[LINK] $@"
	$(CXX) recover.o -o $@ $(LDFLAGS)

# 4. Seed2Priv Tool
$(TARGET_SEED2PRIV): seed2priv_main.o
	@echo "[LINK] $@"
	$(CXX) seed2priv_main.o -o $@ $(LDFLAGS)

# --- REGULI GENERALE DE COMPILARE ---

%.o: %.cpp
	@echo "[CXX]  $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	@echo "[NVCC] $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --- CURĂȚARE ---
clean:
	rm -f *.o $(TARGET_APP) $(TARGET_BLOOM) $(TARGET_RECOVER) $(TARGET_SEED2PRIV)

.PHONY: all clean