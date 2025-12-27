#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

// Includem OpenCL din vcpkg
#include <CL/cl.h>

class OpenCLProvider {
private:
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem d_output = nullptr;
    
    // Kernel-ul OpenCL (RNG Xorshift128+) scris ca string
    // Genereaza 16 bytes de entropie per thread
    const char* kernelSource = R"(
        // Xorshift128+ state structure
        typedef struct {
            ulong s[2];
        } rng_state;

        ulong next(rng_state *state) {
            ulong s1 = state->s[0];
            const ulong s0 = state->s[1];
            state->s[0] = s0;
            s1 ^= s1 << 23;
            state->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
            return state->s[1] + s0;
        }

        __kernel void generate_entropy(__global uchar* output, ulong seed, int count) {
            int gid = get_global_id(0);
            if (gid >= count) return;

            // Initializare seed unic per thread
            rng_state state;
            // Splitmix64 pentru initializare robusta
            ulong z = (seed + gid * 0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
            state.s[0] = z ^ (z >> 31);
            
            z = (state.s[0] + 0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
            state.s[1] = z ^ (z >> 31);

            // Generam 4 numere uint (32bit) = 16 bytes
            uint r1 = (uint)next(&state);
            uint r2 = (uint)next(&state);
            uint r3 = (uint)next(&state);
            uint r4 = (uint)next(&state);

            // Scriem in memorie (offset 16 bytes)
            int offset = gid * 16;
            
            output[offset + 0] = r1 & 0xFF; output[offset + 1] = (r1 >> 8) & 0xFF;
            output[offset + 2] = (r1 >> 16) & 0xFF; output[offset + 3] = (r1 >> 24) & 0xFF;

            output[offset + 4] = r2 & 0xFF; output[offset + 5] = (r2 >> 8) & 0xFF;
            output[offset + 6] = (r2 >> 16) & 0xFF; output[offset + 7] = (r2 >> 24) & 0xFF;

            output[offset + 8] = r3 & 0xFF; output[offset + 9] = (r3 >> 8) & 0xFF;
            output[offset + 10] = (r3 >> 16) & 0xFF; output[offset + 11] = (r3 >> 24) & 0xFF;

            output[offset + 12] = r4 & 0xFF; output[offset + 13] = (r4 >> 8) & 0xFF;
            output[offset + 14] = (r4 >> 16) & 0xFF; output[offset + 15] = (r4 >> 24) & 0xFF;
        }
    )";

    void checkErr(cl_int err, const char* name) {
        if (err != CL_SUCCESS) {
            std::cerr << "[OpenCL ERROR] " << name << " (" << err << ")\n";
            exit(1);
        }
    }

public:
    OpenCLProvider(int platformId = 0, int deviceId = 0) {
        cl_int err;
        
        // 1. Obtine Platforme (AMD, Intel, NVIDIA)
        cl_uint numPlatforms;
        err = clGetPlatformIDs(0, nullptr, &numPlatforms);
        if (numPlatforms == 0) { std::cerr << "Nu s-au gasit platforme OpenCL!\n"; exit(1); }
        
        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        
        if (platformId >= numPlatforms) platformId = 0;
        
        // 2. Obtine Device
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
        
        if (deviceId >= numDevices) deviceId = 0;
        cl_device_id device = devices[deviceId];

        // Info Device
        char devName[128];
        clGetDeviceInfo(device, CL_DEVICE_NAME, 128, devName, nullptr);
        std::cout << "[OpenCL] Initialized: " << devName << "\n";

        // 3. Context & Queue
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        checkErr(err, "Context");
        
        // OpenCL 2.0+ deprecated clCreateCommandQueue, dar e inca suportat. Folosim varianta moderna daca e posibil
        #ifdef CL_VERSION_2_0
        queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
        #else
        queue = clCreateCommandQueue(context, device, 0, &err);
        #endif
        checkErr(err, "Queue");

        // 4. Build Kernel
        program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "[OpenCL Build Error]: " << log.data() << "\n";
            exit(1);
        }

        kernel = clCreateKernel(program, "generate_entropy", &err);
        checkErr(err, "Kernel");
    }

    ~OpenCLProvider() {
        if (d_output) clReleaseMemObject(d_output);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    void allocate(size_t sizeBytes) {
        cl_int err;
        if (d_output) clReleaseMemObject(d_output);
        d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeBytes, nullptr, &err);
        checkErr(err, "Allocate Buffer");
    }

    void generate(unsigned char* hostBuffer, int count, unsigned long long seed) {
        cl_int err;
        
        // Set Arguments
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_output);
        err |= clSetKernelArg(kernel, 1, sizeof(ulong), &seed);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &count);
        checkErr(err, "SetArgs");

        // Execute (Global Work Size)
        size_t globalSize = count;
        // Local size NULL = lasam driverul sa decida optimizarea
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        checkErr(err, "Execute");

        // Read Back
        err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, count * 16, hostBuffer, 0, nullptr, nullptr);
        checkErr(err, "ReadBuffer");
    }
};