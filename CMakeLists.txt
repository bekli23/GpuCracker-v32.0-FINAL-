cmake_minimum_required(VERSION 3.18)
project(GpuCracker LANGUAGES CXX CUDA)

# --- CONFIGURARE COMPILATOR ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- DEPENDINȚE (OpenSSL, OpenCL, SECP256K1) ---
find_package(OpenSSL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Pe Linux, OpenCL este de obicei in sistem. Pe Windows cautam in CUDA toolkit sau VCPKG.
if(UNIX)
    find_library(OPENCL_LIB OpenCL)
else()
    set(OPENCL_LIB "OpenCL.lib") 
endif()

# --- SURSE ---
set(SOURCES
    main.cu
    kernel_ops.cu
    GpuCore.cu
    cuda_provider.cu
    # Header files (doar pentru IDE)
    runner.h
    args.h
    gpu_interface.h
    bloom.h
    utils.h
    cuda_provider.h
    opencl_provider.h
    akm.h
)

add_executable(GpuCracker ${SOURCES})

# --- INCLUDE DIRECTORIES ---
target_include_directories(GpuCracker PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    # Adauga calea catre secp256k1 include daca nu e instalat global
)

# --- LINK LIBRARIES ---
target_link_libraries(GpuCracker PRIVATE 
    CUDA::cudart
    CUDA::curand
    OpenSSL::SSL
    OpenSSL::Crypto
    ${OPENCL_LIB}
    secp256k1 # Asigura-te ca ai libsecp256k1 instalat sau compilat
)

# --- ARHITECTURI GPU (UNIVERSAL) ---
# Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper
set_target_properties(GpuCracker PROPERTIES 
    CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90"
)

# --- FLAGURI SPECIFICE OS ---
if(UNIX)
    target_compile_options(GpuCracker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
    target_link_options(GpuCracker PRIVATE -fopenmp)
else()
    target_compile_options(GpuCracker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
endif()