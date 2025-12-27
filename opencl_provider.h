#pragma once
#include "gpu_interface.h"
#include "utils.h"
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstring> // pentru memcpy

// Link OpenCL automat pentru Visual Studio
#pragma comment(lib, "OpenCL.lib")

class OpenClProvider : public IGpuProvider {
private:
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem d_output = nullptr;
    
    int platformId;
    int deviceId;
    int totalThreads;
    int points;
    int batchSize;
    bool sequentialMode; 
    std::string deviceName;

    // --- KERNEL OPTIMIZAT (Xoshiro256** Vectorizat) ---
    // Genereaza entropie folosind tipuri vectoriale pentru a maximiza latimea de banda
    const char* kernelSource = R"(
        typedef ulong u64;
        typedef uint  u32;

        // Xoshiro256** state rotation
        inline u64 rotl(const u64 x, int k) {
            return (x << k) | (x >> (64 - k));
        }

        // SplitMix64 pentru initializarea seed-ului
        inline u64 splitmix64(u64 *x) {
            u64 z = (*x += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        }

        // Xoshiro256** next
        inline u64 next(u64 *s) {
            const u64 result = rotl(s[1] * 5, 7) * 9;
            const u64 t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl(s[3], 45);
            return result;
        }

        inline u32 swap_endian(u32 val) {
            return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) |
                   ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000);
        }

        __kernel void generate_entropy(__global uchar* output, u64 baseSeed, int points, int useSequential, int entropyBytes) {
            int gid = get_global_id(0);
            
            // State local pentru RNG
            u64 s[4];
            
            // Initializare PRNG (doar daca e random mode)
            if (useSequential == 0) {
                u64 seed = baseSeed + gid; 
                s[0] = splitmix64(&seed);
                s[1] = splitmix64(&seed);
                s[2] = splitmix64(&seed);
                s[3] = splitmix64(&seed);
            }

            int counterOffsetWords = (entropyBytes / 4) - 2;
            if (counterOffsetWords < 0) counterOffsetWords = 0;

            // Procesam 'points' iteratii per thread
            for (int i = 0; i < points; i++) {
                // Calcul index global in buffer-ul de iesire
                // Layout: [Thread0_P0][Thread1_P0]... (Coalesced access pattern preferred)
                // Dar structura actuala cere [Thread0_P0...Pn][Thread1_P0...Pn]
                // Mergem pe structura liniara ceruta de CPU: offset = (gid * points + i) * 32
                
                int outIdx = (gid * points) + i;
                int byteOffset = outIdx * 32;

                if (useSequential == 1) {
                    u64 uniqueId = baseSeed + outIdx;
                    // Scriem direct in global memory byte cu byte sau int cu int
                    // Optimizare: folosim vectori uint8 daca e posibil, dar aici scriem manual
                    
                    __global u32* outPtr = (__global u32*)(output + byteOffset);
                    
                    // Umplem cu zero
                    #pragma unroll
                    for(int k=0; k<8; k++) outPtr[k] = 0;

                    // Punem counterul la final
                    outPtr[counterOffsetWords]     = swap_endian((u32)(uniqueId >> 32));
                    outPtr[counterOffsetWords + 1] = swap_endian((u32)(uniqueId & 0xFFFFFFFF));

                } else {
                    // Generam 256 biti (4 x u64)
                    u64 r1 = next(s);
                    u64 r2 = next(s);
                    u64 r3 = next(s);
                    u64 r4 = next(s);

                    // Scriem vectorizat daca pointerul e aliniat, altfel manual
                    // output e uchar*, cast la ulong* poate fi riscant daca nu e aliniat
                    // dar malloc-ul OpenCL e de obicei aliniat la 4KB.
                    
                    __global ulong* outPtrL = (__global ulong*)(output + byteOffset);
                    outPtrL[0] = r1;
                    outPtrL[1] = r2;
                    outPtrL[2] = r3;
                    outPtrL[3] = r4;
                }
            }
        }
    )";

    void checkErr(cl_int err, const char* name) {
        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL ERROR] " << name << " (" << err << ")\n";
            // Nu dam exit violent, incercam sa continuam sau sa afisam eroarea
        }
    }

public:
    OpenClProvider(int pId, int dId, int threads, int pts, bool seq) 
        : platformId(pId), deviceId(dId), totalThreads(threads), points(pts), sequentialMode(seq) {
        
        // Aliniere threads la multiplu de 64 (Wavefront/Warp)
        if (totalThreads % 64 != 0) totalThreads = ((totalThreads + 63) / 64) * 64;
        batchSize = totalThreads * points;
    }

    ~OpenClProvider() {
        if (d_output) clReleaseMemObject(d_output);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    void init() override {
        cl_int err;
        cl_uint numPlatforms; 
        clGetPlatformIDs(0, nullptr, &numPlatforms);
        
        if (numPlatforms == 0) {
            std::cerr << "[OpenCL] No platforms found!\n";
            return;
        }

        std::vector<cl_platform_id> platforms(numPlatforms); 
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        
        // Logica de selectie inteligenta a platformei (NVIDIA > AMD > INTEL)
        int selectedPlat = -1;
        
        // 1. Cautam platforma ceruta explicit
        if (platformId < (int)numPlatforms) {
            selectedPlat = platformId;
        }

        // 2. Daca e auto sau default, cautam cea mai buna
        if (platformId == 0) {
            for(int i=0; i<(int)numPlatforms; ++i) {
                char pName[128]; 
                clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, pName, nullptr);
                std::string sName = pName;
                // Prioritate NVIDIA
                if (sName.find("NVIDIA") != std::string::npos) { selectedPlat = i; break; }
                // Apoi AMD
                if (sName.find("AMD") != std::string::npos && selectedPlat == -1) { selectedPlat = i; }
            }
            if (selectedPlat == -1) selectedPlat = 0; // Fallback
        }

        cl_uint numDevices; 
        clGetDeviceIDs(platforms[selectedPlat], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        
        // Fallback la CPU daca nu sunt GPU-uri
        if (numDevices == 0) {
             clGetDeviceIDs(platforms[selectedPlat], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        }

        if (numDevices == 0) {
            std::cerr << "[OpenCL] No devices found on platform " << selectedPlat << "\n";
            return;
        }

        std::vector<cl_device_id> devices(numDevices); 
        clGetDeviceIDs(platforms[selectedPlat], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
        
        if (deviceId >= (int)numDevices) deviceId = 0;
        cl_device_id dev = devices[deviceId];
        
        char name[128]; 
        clGetDeviceInfo(dev, CL_DEVICE_NAME, 128, name, nullptr);
        deviceName = "[OpenCL] " + std::string(name);

        // Creare Context
        context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
        checkErr(err, "Context");
        
        // Creare Queue (In-order is fine for basic kernel)
        #ifdef CL_VERSION_2_0
            cl_queue_properties props[] = {0}; 
            queue = clCreateCommandQueueWithProperties(context, dev, props, &err);
        #else
            queue = clCreateCommandQueue(context, dev, 0, &err);
        #endif
        checkErr(err, "Queue");

        // Compilare Program
        program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
        const char* buildOptions = "-cl-fast-relaxed-math -cl-mad-enable";
        err = clBuildProgram(program, 1, &dev, buildOptions, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            size_t len; clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            std::vector<char> buffer(len + 1);
            clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, len, buffer.data(), NULL);
            std::cerr << "[OpenCL Build Error]: " << buffer.data() << "\n";
            return; // Stop init
        }

        kernel = clCreateKernel(program, "generate_entropy", &err);
        checkErr(err, "Kernel Create");
        
        // --- ALOCARE MEMORIE PINNED (ZERO COPY DACA E POSIBIL) ---
        // CL_MEM_ALLOC_HOST_PTR forteaza driverul sa aloce memorie accesibila rapid de host
        d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)batchSize * 32, nullptr, &err);
        checkErr(err, "Buffer Alloc");
    }

    void generate(unsigned char* hostBuffer, unsigned long long seed, int entropyBytes) override {
        cl_int err;
        int useSeqInt = sequentialMode ? 1 : 0;
        
        // Set Args
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_output);
        clSetKernelArg(kernel, 1, sizeof(unsigned long long), &seed);
        clSetKernelArg(kernel, 2, sizeof(int), &points);
        clSetKernelArg(kernel, 3, sizeof(int), &useSeqInt);
        clSetKernelArg(kernel, 4, sizeof(int), &entropyBytes); 
        
        size_t globalSize = totalThreads;
        size_t localSize = 64; // Tuning pentru majoritatea GPU-urilor

        // Lansare Kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
        if(err != CL_SUCCESS) {
             // Fallback daca localSize e invalid pt device
             err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        }
        // checkErr(err, "EnqueueNDRangeKernel"); // Nu vrem sa crpam in loop

        // --- MAP BUFFER PENTRU VITEZA ---
        // Maparea evita copierea implicita prin driver si permite acces direct la memoria pinned
        void* ptr = clEnqueueMapBuffer(queue, d_output, CL_TRUE, CL_MAP_READ, 0, (size_t)batchSize * 32, 0, nullptr, nullptr, &err);
        
        if (err == CL_SUCCESS && ptr) {
            // Copiem datele in bufferul aplicatiei
            std::memcpy(hostBuffer, ptr, (size_t)batchSize * 32);
            
            // Unmap
            clEnqueueUnmapMemObject(queue, d_output, ptr, 0, nullptr, nullptr);
        } else {
             // Fallback ReadBuffer
             clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, (size_t)batchSize * 32, hostBuffer, 0, nullptr, nullptr);
        }
    }

    int getBatchSize() const override { return batchSize; }
    std::string getName() const override { return deviceName; }
    std::string getConfig() const override { return "Th:" + std::to_string(totalThreads) + " Pts:" + std::to_string(points) + " [OpenCL]"; }
};