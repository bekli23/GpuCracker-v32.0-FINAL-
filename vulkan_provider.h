#pragma once
#include "gpu_interface.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#pragma comment(lib, "vulkan-1.lib")
#endif

#define VK_CHECK(result) \
    if (result != VK_SUCCESS) { \
        std::cerr << "[Vulkan ERR] Code: " << result << " line: " << __LINE__ << "\n"; \
        throw std::runtime_error("Vulkan Error"); \
    }

class VulkanProvider : public IGpuProvider {
private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;

    VkBuffer outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory outputBufferMemory = VK_NULL_HANDLE;

    int deviceIndex;
    int totalThreads;
    int points;
    int batchSize;
    bool sequentialMode;
    std::string deviceName;

    struct PushConsts {
        uint64_t baseSeed;
        uint32_t points;
        uint32_t useSequential;
        uint32_t entropyBytes;
    };

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) return {};
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Vulkan: Failed to find suitable memory type!");
    }

public:
    VulkanProvider(int dIdx, int threads, int pts, bool seq) 
        : deviceIndex(dIdx), totalThreads(threads), points(pts), sequentialMode(seq) {
        batchSize = totalThreads * points;
    }

    ~VulkanProvider() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (outputBuffer) vkDestroyBuffer(device, outputBuffer, nullptr);
            if (outputBufferMemory) vkFreeMemory(device, outputBufferMemory, nullptr);
            if (shaderModule) vkDestroyShaderModule(device, shaderModule, nullptr);
            if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            if (descriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    void init() override {
        // 1. Instance
        VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.pApplicationName = "GpuCracker";
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        createInfo.pApplicationInfo = &appInfo;
        VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

        // 2. Physical Device
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) throw std::runtime_error("No Vulkan GPUs found");
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        if (deviceIndex >= (int)deviceCount) deviceIndex = 0;
        physicalDevice = devices[deviceIndex];

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        deviceName = "[Vulkan] " + std::string(props.deviceName);

        // 3. Queue Family
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        bool found = false;
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("No compute queue found");

        // 4. Logical Device
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        
        VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);

        // 5. Buffer (Output)
        VkDeviceSize bufferSize = (VkDeviceSize)batchSize * 32;
        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &outputBuffer));

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, outputBuffer, &memReqs);
        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &outputBufferMemory));
        VK_CHECK(vkBindBufferMemory(device, outputBuffer, outputBufferMemory, 0));

        // 6. Descriptors
        VkDescriptorSetLayoutBinding layoutBinding = {};
        layoutBinding.binding = 0;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &layoutBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));

        VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = descriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &descriptorSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &setAllocInfo, &descriptorSet));

        VkDescriptorBufferInfo dBufferInfo = {};
        dBufferInfo.buffer = outputBuffer;
        dBufferInfo.offset = 0;
        dBufferInfo.range = bufferSize;

        VkWriteDescriptorSet descriptorWrite = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &dBufferInfo;
        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

        // 7. Pipeline & Shader
        auto shaderCode = readFile("akm_vulkan.spv");
        if (shaderCode.empty()) throw std::runtime_error("Failed to load akm_vulkan.spv! Compile shader first!");

        VkShaderModuleCreateInfo shaderInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        shaderInfo.codeSize = shaderCode.size();
        shaderInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
        VK_CHECK(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule));

        VkPushConstantRange pushRange = {};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConsts);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

        VkComputePipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

        // 8. Command Buffer & Fence
        VkCommandPoolCreateInfo poolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device, &poolCreateInfo, nullptr, &commandPool));

        VkCommandBufferAllocateInfo cmdAllocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer));

        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    }

    void generate(unsigned char* hostBuffer, unsigned long long seed, int entropyBytes) override {
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &fence);
        vkResetCommandBuffer(commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        PushConsts pc;
        pc.baseSeed = seed;
        pc.points = (uint32_t)points;
        pc.useSequential = sequentialMode ? 1 : 0;
        pc.entropyBytes = (uint32_t)entropyBytes;
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts), &pc);

        uint32_t groups = (totalThreads + 63) / 64;
        vkCmdDispatch(commandBuffer, groups, 1, 1);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, fence));

        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        void* mappedData;
        vkMapMemory(device, outputBufferMemory, 0, (VkDeviceSize)batchSize * 32, 0, &mappedData);
        memcpy(hostBuffer, mappedData, (size_t)batchSize * 32);
        vkUnmapMemory(device, outputBufferMemory);
    }

    int getBatchSize() const override { return batchSize; }
    std::string getName() const override { return deviceName; }
    std::string getConfig() const override { return "Th:" + std::to_string(totalThreads) + " [Vulkan]"; }
};