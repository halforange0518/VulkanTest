#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream> // 用于输出信息
#include <stdexcept> // 用于报告错误
#include <cstdlib> // 用于提供 EXIT_SUCCESS 和 EXIT_FAILURE 宏
#include <vector>
#include <cstring>
#include <map>
#include <set>
#include <optional>
#include <cstdint> // 用于提供 uint32_t
#include <limits> // 用于提供 std::numeric_limits
#include <algorithm> // 用于提供 std::clamp
#include <fstream> // 用于读取文件

const uint32_t WIDTH = 800;// 窗口的宽和高
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;// 表明同时可以处理多少帧

// validationLayers存储了所有申请的验证层，通过指定其名称来启用，这里启用了LunarG提供的标准验证层（类似于STL）
const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
};

// 声明显卡所需的扩展：交换链扩展
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// 只有在debug情况下才启用验证层，Release不启用
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// 创建信息回调对象
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    // 使用vkGetInstanceProcAddr来查找地址
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// 销毁信息回调对象
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// 队列族
// 在Vulkan中几乎所有操作 (从绘制到上传纹理) 都需要将命令提交到队列，有不同类型的队列，它们来自不同的队列族
struct QueueFamilyIndices {
    // optional是一个包装器，它在初始化时不会分配任何值，此时has_value() == false，只有当手动分配值时，has_value() == true
    std::optional<uint32_t> graphicsFamily;// 绘图命令队列
    std::optional<uint32_t> presentFamily;// 表示队列：支持图像呈现到表面的队列

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

// 交换链信息细节，由于数据比较多，用结构体储存
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;// 基本表面（surface）功能
    std::vector<VkSurfaceFormatKHR> formats;// 支持的表面格式
    std::vector<VkPresentModeKHR> presentModes;// 可用的呈现（presentation）模式
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;// 窗口
    VkInstance instance;// 实例
    VkDebugUtilsMessengerEXT debugMessenger;// 消息回调
    VkSurfaceKHR surface;// 表面/界面对象

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;// 物理设备
    VkDevice device;// 逻辑设备

    VkQueue graphicsQueue;// 图形队列句柄，其保存了逻辑设备与队列之间的接口
    VkQueue presentQueue;// 表示队列句柄

    VkSwapchainKHR swapChain;// 交换链
    std::vector<VkImage> swapChainImages;// 交换链中的图像，它与交换链绑定，一旦交换链被删除，它也会被删除
    VkFormat swapChainImageFormat;// 交换链图像的格式
    VkExtent2D swapChainExtent;// 交换链图像的范围
    std::vector<VkImageView> swapChainImageViews;// 图像视图

    VkRenderPass renderPass;// 渲染通道
    VkPipelineLayout pipelineLayout;// 管道布局，用来指定shader中的uniform值
    VkPipeline graphicsPipeline;// 图形管道

    std::vector<VkFramebuffer> swapChainFramebuffers;// 帧缓冲

    VkCommandPool commandPool;// 命令池
    std::vector<VkCommandBuffer> commandBuffers;// 命令缓冲，其与命令池绑定，命令池删除，它也会被删除（一帧对应一个缓冲，多帧即为数组）

    // 创建同步对象
    std::vector<VkSemaphore> imageAvailableSemaphores;// 信号量：表示已经从交换链中获取了图像并准备好渲染（一帧对应一个信号量，多帧即为数组）
    std::vector<VkSemaphore> renderFinishedSemaphores;// 信号量：表示渲染已经完成并且可以进行呈现（一帧对应一个信号量，多帧即为数组）
    std::vector<VkFence> inFlightFences;// 栅栏：确保一个循环只渲染一帧（一帧对应一个栅栏，多帧即为数组）

    bool framebufferResized = false;// 记录窗口图像大小是否被修改

    uint32_t currentFrame = 0;// 帧的索引，用来指示当前绘制的是哪一帧

    // 初始化窗口
    void initWindow() {
        glfwInit(); // glfw初始化

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // glfw默认会创建OpenGL环境，手动选择不创建
        // glfw是默认开启窗口拉伸的，如果禁用，需要用下面的语句进行禁用
        //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // 暂时禁用窗口拉伸

        // 函数结构：创建窗口（窗口高度，窗口宽度，窗口标题，指定显示器（nullptr就是默认显示器），此选项只与OpenGL有关）
        window = glfwCreateWindow(WIDTH, HEIGHT, "VulkanTest", nullptr, nullptr);
        // 将当前窗口与检查窗口大小修改的回调函数进行绑定，一旦窗口大小修改，回调函数就会被调用
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    // 检查窗口帧大小是否改变的回调函数
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    // 初始化Vulkan
    void initVulkan() {
        createInstance();// 创建实例
        setupDebugMessenger();// 创建消息回调
        // 注意！消息回调是在实例创建之后才创建的，而且在实例删除之前消息回调就已经删除了，这会导致在实例创建和删除时，无法进行消息回调
        // 为了能实现此部分的消息回调，我们可以在创建实例指定验证层时单独加上一个消息回调信使，详见创建实例处代码
        createSurface();// 创建表面对象绑定窗口
        pickPhysicalDevice();// 寻找并选择物理设备：图形GPU
        createLogicalDevice();// 创建逻辑设备
        createSwapChain();// 创建交换链
        createImageViews();// 创建图像视图
        createRenderPass();// 渲染通道
        createGraphicsPipeline();// 创建图形管线
        createFramebuffers();// 创建帧缓冲
        createCommandPool();// 创建命令池
        createCommandBuffers();// 创建命令缓冲
        createSyncObjects();// 创建同步对象
    }

    // 主循环
    void mainLoop() {
        // 每个循环检测是否有关闭窗口命令，有则结束主循环
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        // 由于drawFrame函数中的操作是异步的，直接跳出循环会导致其中的某些步骤还在执行，而此时进行资源删除会导致出错
        vkDeviceWaitIdle(device);// 这里使用vkDeviceWaitIdle等待命令队列的操作完成
    }

    // 主循环结束之后执行清理函数，用于释放资源
    // 创建和删除的顺序满足：先创建的后删除，后创建的先删除
    void cleanup() {
        cleanupSwapChain();// 删除交换链，以及与之绑定的交换链图像视图和交换链帧缓冲

        vkDestroyPipeline(device, graphicsPipeline, nullptr);// 删除图形管道
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);// 删除管道布局

        vkDestroyRenderPass(device, renderPass, nullptr);// 删除渲染通道

        // 删除同步对象
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);// 删除命令池

        vkDestroyDevice(device, nullptr);// 删除逻辑设备

        // 删除验证层的消息回调
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);// 删除表面对象
        vkDestroyInstance(instance, nullptr); // 删除实例

        glfwDestroyWindow(window); // 删除窗口

        glfwTerminate();
    }

    // 创建实例，实例用于给驱动程序指定一些此应用的信息
    void createInstance() {
        // 检查申请的验证层是否都可用
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }
        // Vulkan中的许多信息是通过结构而不是函数参数传递的，所以需要建立一些结构体来指定信息

        VkApplicationInfo appInfo{}; // 此应用的一些信息
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{}; // 实例创建信息
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo; // 之前声明的信息（appInfo）已经嵌套于此

        // 获得扩展
        auto extensions = getRequiredExtensions();

        // 如果使用MacOS + MoltenVK，会出现 VK_ERROR_INCOMPATIBLE_DRIVER 错误，需要进行额外操作
#ifdef __MACH__
        // MoltenVK兼容性扩展
        extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        extensions.emplace_back("VK_KHR_get_physical_device_properties2");

        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        // 指定扩展的数量和数据
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // 指定验证层的数量和数据，顺带为其创建消息回调信使，信使在创建和删除实例时都可以使用
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // 创建消息回调
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        // 信息指定完即可创建实例，vkCreateInstance会返回VkResult类型的结果，如果创建成功，会返回VK_SUCCESS
        // 函数结构：创建实例（创建信息指针，回调函数指针，实例指针）
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    // 验证层检查函数，用来检查请求的验证层是否可用，如果返回false，说明申请了不可用的验证层
    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        // 第一次执行此函数会得到所有可用层的数量，并赋值给layerCount
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        // 第二次执行此函数会把所有可用层存入availableLayers中
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // 接下来就是检查请求的验证层是否都在availableLayers中，不在就返回false
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                // strcmp用来比较两个字符串是否一致
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) return false;
        }
        return true;
    }

    std::vector<const char*> getRequiredExtensions() {
        // Vulkan是一个与平台无关的API，所以需要一个扩展（Extension）来连接Vulkan与系统窗口
        // glfw正好提供了这么一个扩展
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        // glfw提供的扩展
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // 额外添加的扩展
        // 验证层的消息回调扩展，VK_EXT_DEBUG_UTILS_EXTENSION_NAME是宏定义，等于字符串"VK_EXT_debug_utils"
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    // 回调函数，其中有四个参数，关于参数含义请参考vulkan-tutorial的Validation layers项
    // 其中最后一项pUserData参数包含一个指针，该指针在设置回调期间指定，并允许将自己的数据传递给它
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData) {

        // messageSeverity的信息是有严重程度的，可以根据类型输出不同消息，严重程度由小到大分别为：
        // 1.诊断信息 2.类似创建资源的普通信息 3.警告信息（可能是应用程序错误） 4.错误信息（行为无效并可能导致崩溃）
        if (messageSeverity > VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            std::cerr << "validation layer ERROR: " << pCallbackData->pMessage << std::endl;
        } else if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            std::cerr << "validation layer WARNING: " << pCallbackData->pMessage << std::endl;
        } else {
            std::cout << "validation layer MESSAGE: " << pCallbackData->pMessage << std::endl;
        }

        return VK_FALSE;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        // 跟实例一样，需要建立一个结构体来指定相关信息
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);// 因为需要复用，所以写成函数
        createInfo.pUserData = nullptr; // 可选项

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }

    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    // 创建表面对象并绑定窗口
    void createSurface() {
        // 直接使用GLFW自带的glfwCreateWindowSurface的方法将表面对象与窗口进行绑定，这样可以处理平台差异
        // 这种方法不需要建立结构体，只需要简单的传入参数
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    // 查找并选择合适的GPU
    void pickPhysicalDevice() {
        // 显卡是可以选择多个的，但这里我们只选择一个，用physicalDevice记录最终选择的显卡

        // 检查可用的显卡数量
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        // 如果不存在可用显卡，直接报错
        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        // 用数组devices存储所有可用显卡
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // 并不是每个可用显卡都是合适的显卡，我们需要找到满足一定条件的适用性显卡

        // 可以用分数对显卡进行衡量，选择分数最大的显卡（这里代码只作参考）
//        std::multimap<int, VkPhysicalDevice> candidates;
//
//        for (const auto& device : devices) {
//            int score = rateDeviceSuitability(device);
//            std::cout << "graphics card score:" << score << std::endl;
//            candidates.insert(std::make_pair(score, device));
//        }
//
//        // 只有分数大于0的显卡才是满足条件的显卡，由于用map进行了排序，排在第一个的就是分数最大的显卡
//        if (candidates.rbegin()->first > 0) {
//            physicalDevice = candidates.rbegin()->second;
//        } else {
//            throw std::runtime_error("failed to find a suitable GPU!");
//        }

        // 这里找到一张合适显卡就返回
        for (const auto& _device : devices) {
            if (isDeviceSuitable(_device)) {
                physicalDevice = _device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    // 根据显卡的条件进行打分，这个方法暂时不用，只是用作参考
    int rateDeviceSuitability(VkPhysicalDevice _device) {
        // 查询设备基本属性，如：名称、类型、支持的Vulkan版本
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(_device, &deviceProperties);
        // 查询对可选功能的支持，如：纹理压缩、64位浮点数、多视角渲染（对VR有用）
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(_device, &deviceFeatures);

        // 根据自定条件给显卡打分
        int score = 0;

        // 尽量选择离散GPU，它有显著的性能优势
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        }

        // 查询显卡的最大可绘制纹理，其很大程度影响图形质量
        score += deviceProperties.limits.maxImageDimension2D;

        // 查询是否支持几何着色器，如果不支持，程序将无法运行
        if (!deviceFeatures.geometryShader) {
            return 0;
        }

        return score;
    }

    // 检查显卡是否合适
    bool isDeviceSuitable(VkPhysicalDevice device) {
        // 检查显卡是否支持指定的队列
        QueueFamilyIndices indices = findQueueFamilies(device);
        // 检查显卡是否支持指定的设备扩展
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        // 检查显卡对于交换链细节的支持
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            // 只要至少有一种受支持的表面格式和一种受支持的呈现模式，此交换链就视为支持
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    // 寻找设备支持的队列族
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice _device) {
        QueueFamilyIndices indices;

        // 寻找物理设备支持的队列族
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(_device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(_device, &queueFamilyCount, queueFamilies.data());

        // 寻找至少一个支持VK_QUEUE_GRAPHICS_BIT的队列族
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            // 寻找支持表面表示的队列
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(_device, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }
            // 只要需求的所有队列找到了，就直接返回这个族
            if (indices.isComplete()) {
                break;
            }
            i++;
        }

        return indices;
    }

    // 创建逻辑设备
    void createLogicalDevice() {
        // 跟创建实例一样，需要创建结构体将信息进行指定

        // 指定要创建的队列
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        // 用vector来存储多个队列族
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        // 用set来存储需求的队列：图形队列、表示队列
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        // 需要给每个队列指定优先级（范围为0.0-1.0），它将影响命令缓冲区执行的调度顺序
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // 指定设备的功能集，目前我们不需要额外的功能，所以此项置空
        VkPhysicalDeviceFeatures deviceFeatures{};

        // 创建逻辑设备
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        // 指定队列和功能集
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;

        // 跟创建实例一样，需要指定扩展和验证层（存在一些设备独有的扩展和验证层）

        // 指定扩展，需要加入的扩展已经放入deviceExtensions中了
        std::vector<const char*> deviceExtensionsAll(deviceExtensions.begin(), deviceExtensions.end());
        // MacOS需要一个额外的扩展VK_KHR_portability_subset来兼容MoltenVK
#ifdef __MACH__
        deviceExtensionsAll.push_back("VK_KHR_portability_subset");
#endif

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensionsAll.size());
        createInfo.ppEnabledExtensionNames = deviceExtensionsAll.data();

        // 指定验证层，之前设置的验证层依旧适用
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        // 将物理设备与新建的逻辑设备相连
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // 根据逻辑设备和队列，获得图形队列句柄graphicsQueue，其保存了两者之间的接口
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        // 表示队列句柄presentQueue
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // 检查显卡是否有指定的扩展
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        // 获得此显卡支持的所有扩展
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        // 检查此显卡支持的所有扩展中是否有所需的扩展
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    // 获得数据链信息
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;
        // 获得基本表面功能
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        // 获得支持的表面格式
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        // 获得可用的呈现模式
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    // 从所有支持的表面格式中选择一个最理想的
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // 每个VkSurfaceFormatKHR条目都包含一个格式（format）和一个颜色空间（colorSpace）成员。
        // 格式指定颜色通道和类型
        // 颜色空间指示SRGB颜色空间是否支持或不使用VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        // 对于颜色空间，我们将使用SRGB，因为它有更准确的感知颜色，它也几乎是图像的标准颜色空间，最常见的SRGB为VK_FORMAT_B8G8R8A8_SRGB

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        // 如果不满足条件，就用第一种格式即可
        return availableFormats[0];
    }

    // 从所有支持的呈现模式中选择一个最理想的
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // 有四种呈现模式（详细说明请看笔记）：
        // 1、立即模式（VK_PRESENT_MODE_IMMEDIATE_KHR） 2、缓冲模式（VK_PRESENT_MODE_FIFO_KHR）
        // 3、立即缓冲模式（VK_PRESENT_MODE_FIFO_RELAXED_KHR） 4、三重缓冲模式（VK_PRESENT_MODE_MAILBOX_KHR）
        // 我们这里选择第四个：三重缓冲模式作为目标，三重缓冲的效果最好，但是花费较高
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        // 如果没有三重缓冲，就选择第二个：缓冲模式
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // 指定正确的交换范围
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // 交换范围就是交换链图像的分辨率，它几乎总是完全等于绘制的窗口的分辨率
        // 我们使用glfwGetFramebufferSize来查询像素窗口的分辨率，然后再将其与最小和最大图像范围进行匹配
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);// 获得以像素为单位的界面分辨率

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            // 使用clamp函数将宽度和高度的值绑定在实现支持的允许最小和最大范围之间
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    // 创建交换链
    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        // 将交换链的各种信息指定为最理想状态
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // 指定交换链存储图像的最小数量（可以更大）
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        // 用结构传递信息
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;// 每个图像组成的图层数量，除非在开发立体3D应用程序（VR），否则这始终是1
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;// 选择直接渲染方式

        // 还需要指定交换链和队列的映射
        // 如果图形队列和呈现队列不属于同一队列族，那我们需要处理多个队列族与交换链映射的情况：我们将从图形队列中绘制交换链中的图像，然后将它们提交到呈现队列中
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily) {
            // 可以跨多个队列族使用图像，而无需显式的所有权转移（用并发模式解决跨队列族问题）
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            // 映射一次由一个队列族拥有，并且在将其用于另一个队列族之前，必须显式地转移所有权（如果在同一个族，就坚持独占模式）
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }
        // 可以指定图像是否需要转换（例如：90度顺时针旋转或水平翻转），这里我们不需要这些转换，只需指定当前转换
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        // 指定窗口是否使用Alpha通道（使用的话，窗口会有透明样式），这里我们不使用
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        // 指定呈现模式（图像缓冲模式）
        createInfo.presentMode = presentMode;
        // 是否启用裁剪，这里我们启用裁剪，表明我们不关心被遮挡的像素的颜色，启用此项可以提高性能
        createInfo.clipped = VK_TRUE;
        // Vulkan在运行中，交换链可能会失效（例如调整窗口大小，分辨率发生改变），这种情况下我们需要重新创建一个交换链
        // 而当新的交换链创建之后，就需要指定对旧交换链的引用
        // 目前我们只创建一个交换链，所以不存在旧交换链，暂时指定为空
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        // 检索交换链图像
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        // 指定图像格式和范围
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    // 创建图像视图
    void createImageViews() {
        // 将视图数量调整为交换链图像的数量，其一一对应
        swapChainImageViews.resize(swapChainImages.size());

        // 创建每个图像视图
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            // 指定图像类型：1D纹理、2D纹理、3D纹理和立方体贴图，这里我们视为2D纹理
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            // 颜色通道映射，这里使用默认映射
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            // subresourceRange字段描述了图像的用途以及应访问图像的哪一部分，这里只是将图像用作普通颜色目标，没有mipmap，也不存在多层
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;// 如果使用立体3D应用程序（VR），那么需要创建具有多层（例如左眼、右眼视图）的交换链
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    // 读取文件数据，并以char数组vector返回
    static std::vector<char> readFile(const std::string& filename) {
        // ate：在文件末尾开始读取（只是为了得到文件的总长度），binary: 将文件以二进制格式进行读取（避免文本转换）
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        // 获得文件长度，并设置数组大小
        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);
        // 返回到文件开头，开始读取数据
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    // 创建图形管线
    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("../shaders/vert.spv");
        auto fragShaderCode = readFile("../shaders/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // 将shader分配给指定的管道阶段（顶点、片段着色阶段等）
        // 先指定顶点着色阶段
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;// 指定shader模块
        vertShaderStageInfo.pName = "main";// 指定要调用模块中的哪个函数，这里指定为main函数
        // 由指定模块函数可知，我们可以将多个片段着色器组合到一个着色器模块中，并使用不同的入口点来区分它们的行为

        // 还有一个 (可选) 成员，pSpecializationInfo，我们不会在这里使用，但值得讨论。
        // 它允许您为着色器常量指定值。您可以使用单个着色器模块，通过为其中使用的常量指定不同的值，可以在管道创建时配置其行为。
        // 这比在渲染时使用变量配置着色器更有效，因为编译器可以进行优化，例如"消除依赖于这些值的if语句"。
        // 如果你没有这样的常量，就不用管，程序会自动将成员设置为nullptr。

        // 片段着色阶段同理
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        // 描述传输给顶点着色器的顶点数据的格式，例如每个顶点的数据间隔，数据是顶点坐标还是实例，偏移量等等
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

        // 指定将从顶点绘制什么样的几何形状（点，线，条带线，面，条带面）以及是否应启用primitiveRestartEnable（高级设置，可以执行重用顶点等）
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // 将视口和裁剪区域设定为动态数据（更加灵活），然后在绘制时设置实际的视口和裁剪矩形
        // 视口范围：视口描述了将输出呈现到的帧缓冲区的区域，一般都是(0,0) 到 (width, height)，这里是交换链图像的的宽和高（交换链图像是和帧缓冲区绑定的），它可能与窗口的宽和高不同
        // 裁剪矩形区域：裁剪矩形之外的任何像素都将被光栅化器丢弃（类似于滤波器），如果不想裁剪，就将范围指定为帧缓冲区大小即可
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // 设置光栅化参数，它从顶点着色器中获取由顶点成形的几何形状，并将其转换为要由片段着色器着色的片段
        // 它还执行深度测试，面剔除和裁剪测试，并且可以将其配置为输出填充整个多边形或仅填充边缘 (线框渲染) 的片段
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;// 是否启用深度测试，启动它需要启用GPU feature
        rasterizer.rasterizerDiscardEnable = VK_FALSE;// 如果此项启用，则几何永远不会经过光栅化阶段，一般不启用
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;// 此项决定几何形状，只有三种：点POINT、线LINE、面FILL，除了面不需要启用GPU feature，其他都要
        rasterizer.lineWidth = 1.0f;// 线宽，其最大值取决于其硬件，设置大于1.0f的线需要启用名为"wideLines"的GPU feature
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;// 面剔除，有四种：禁用剔除，剔除正面、剔除背面、剔除正面和背面
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;// 指定面的正反与点的顺序之间的关系，这里指定顺时针为正面（逆时针为COUNTER_CLOCKWISE）
        rasterizer.depthBiasEnable = VK_FALSE;// 通过面片斜率或者固定值来更改深度值，这里我们不修改
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

        // 多重采样（抗锯齿）
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;// 这里不启用抗锯齿，启用它需要启用GPU feature
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // Optional
        multisampling.pSampleMask = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable = VK_FALSE; // Optional

        // 颜色混合Blending
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE; // 这里不启用混合
        // 如果启用混合，下面是混合用的参数，最常见的是alpha混合：finalColor.rgb = alpha * newColor + (1 - alpha) * oldColor;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
        // 这里指定了帧缓冲的引用，还有用于上述混合公式用的四个参数
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        // 管道布局，用来指定shader中的uniform值（类似于shader中的可以动态改变的全局变量），目前先不指定
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0; // Optional
        pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // 所有前置步骤已经完成，可以创建图形管线了，将之前创建的信息指定给管线
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // Optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1; // Optional

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        // 已经将模组指定完成，模组使命已经完成，用完即删
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    // 将shader代码包装在VkShaderModule对象中
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        // 将代码数据和大小传入结构体中
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        // 创建shaderModule对象
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }

    // 渲染通道，在完成管道创建之前，我们需要告诉Vulkan渲染时将使用的framebuffer附件
    // 我们需要指定将有多少个颜色和深度缓冲区，每个缓冲区要使用多少个样本，以及在整个渲染操作中应如何处理它们的内容
    void createRenderPass() {
        // 这里我们只指定一个颜色缓冲区附件
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;// 颜色附件的格式应该与交换链图像的格式相匹配
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;// 还没有采用多重采样，所以我们使用1个样本
        // loadOp和storeOp决定在渲染之前和渲染之后如何处理附件中的数据（颜色和深度数据）
        // loadOp分为：
        // VK_ATTACHMENT_LOAD_OP_LOAD: 保留附件的现有内容
        // VK_ATTACHMENT_LOAD_OP_CLEAR: 在开始时将值清除为常量
        // VK_ATTACHMENT_LOAD_OP_DONT_CARE: 现有内容未定义，我们不在乎它们
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;// 我们每次绘制帧，都将帧缓冲区清楚为黑色，然后重新绘制帧
        // storeOp分为：
        // VK_ATTACHMENT_STORE_OP_STORE: 渲染的内容将存储在内存中，以后可以读取
        // VK_ATTACHMENT_STORE_OP_DONT_CARE: 渲染操作后，帧缓冲区的内容将设为未定义
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;// 这里选择选择存储数据
        // stencilLoadOp/stencilStoreOp适用于模板数据，我们的应用程序不会对模板缓冲区做任何事情
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // 图像的像素布局（决定纹理和帧缓冲），一般分为三种：
        // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: 用作颜色附件的图像
        // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: 要在交换链中呈现的图像
        // VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: 用作内存复制操作目标的图像
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;// 渲染开始之前的布局
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;// 渲染完成时的布局

        // 单个渲染通道可以由多个子通道组成。子通道是依赖于先前通道中的帧缓冲区内容的后续渲染操作，例如一个接一个地应用的一系列后处理效果。
        // 这里我们只使用一个子通道
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

        // 颜色附件
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;// 附件的索引，由于只有单个数组，所以从0开始，正好指定的是片段shader里的layout(location = 0) out vec4 outColor;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;// 使用附件作为颜色缓冲区

        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;// 指定颜色附件

        // 子通道依赖关系，它指定了子通道之间的内存和执行依赖关系
        // 虽然我们只有一个子通道，但是每帧的前后子通道也是两个关联的子通道
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;// 指定依赖项
        dependency.dstSubpass = 0;// 依赖项的索引
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        // 创建渲染通道
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;// 指定附件
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;// 指定子通道
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    // 创建帧缓冲，帧缓冲是与交换链图像视图绑定的
    void createFramebuffers() {
        // 将帧缓冲的大小和交换链图像视图大小对齐
        swapChainFramebuffers.resize(swapChainImageViews.size());

        // 遍历图像视图并从中创建帧缓冲区
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;// 指定缓冲的渲染通道
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;// 此项表示了将swapChainImageViews作为渲染通道的附件进行了绑定
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    // 创建命令池，Vulkan中的命令 (例如绘图操作和内存传输) 不会直接使用函数调用来执行，必须在命令池中记录要执行的所有操作
    void createCommandPool() {
        // 获得物理设备的队列族
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // 命令池的flag只有两种：
        // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: 提示命令池经常使用新命令重新录制 (可能会更改内存分配行为)
        // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: 允许命令池单独重新录制，没有这个标志，它们都必须一起重置
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;// 我们将每帧记录一个命令池，因此我们希望能够对其进行重置和重新记录
        // 命令池是通过在其中一个设备队列上提交来执行的
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();// 我们将记录用于绘图的命令，所以选择图形队列

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // 创建命令缓冲
    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);// 将命令缓冲数组大小与同时绘制的帧数对齐

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;// 将命令池与命令缓冲绑定
        // 级别参数指定分配的命令缓冲区是主要还是辅助命令缓冲区
        // VK_COMMAND_BUFFER_LEVEL_PRIMARY: 可以提交到队列执行，但不能从其他命令缓冲调用
        // VK_COMMAND_BUFFER_LEVEL_SECONDARY: 不能直接提交，但可以从主要命令缓冲调用
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;// 我们只设置一个主要命令缓冲
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();// 指定缓冲区的数量，一帧对应一个缓冲

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void createSyncObjects() {
        // 创建信号量和栅栏
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);// 数量与帧数对齐
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;// 初始将栅栏设置为已发出信号的状态，这是为了让刚开始绘制的第一帧可以顺利绘制

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {

                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }

    }

    // 将要执行的命令写入命令缓冲区，需要传入命令缓冲和交换链图像索引，此过程称为命令缓冲的记录
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // flags参数指定我们将如何使用命令缓冲区。以下值可用:
        // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: 命令缓冲区在执行一次后就会被重新录制
        // VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: 这是一个辅助命令缓冲区，它将完全在单个渲染过程中
        // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: 命令缓冲区可以重新提交，同时它也已经处于待定执行状态
        beginInfo.flags = 0; // 这里我们不使用以上任何值
        beginInfo.pInheritanceInfo = nullptr; // 此值仅与辅助命令缓冲区相关，它指定从调用主要命令缓冲继承哪个状态

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        // 开始渲染传递，先指定相关数据
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;// 指定渲染通道
        // 指定帧缓冲绑定的附件，使用传入的imageIndex参数，我们可以为当前交换链图像选择正确的帧缓冲区
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        // 接下来的两个参数定义了渲染区域的大小和偏移，该区域之外的像素将具有未定义的值
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;
        // 指定每帧清除时使用的颜色，我们指定为黑色
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
        // 渲染传递，这个函数无论如何返回的都是void，所以无法检查错误
        // 函数的最后一个参数控制如何提供渲染过程中的绘图命令。它可以有两个值之一：
        // VK_SUBPASS_CONTENTS_INLINE: 渲染传递命令将嵌入主要命令缓冲区本身，并且不会执行辅助命令缓冲区
        // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS: 渲染传递命令将从辅助命令缓冲区执行
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // 绑定图形管线，函数第二个参数指定管线对象是图形管线还是计算管线
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // 指定视口和裁剪矩阵的大小
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // 绘制命令，函数结构：（命令缓冲，顶点数量，实例数量，顶点缓冲的偏移量，实例的偏移量）
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        // 结束渲染传递
        vkCmdEndRenderPass(commandBuffer);
        // 结束命令缓冲的记录
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // 绘制帧，当绘制多帧时，要根据帧索引currentFrame来引用数据数组对应的下标
    void drawFrame() {
        // 1、等待上一帧绘制完成
        // 利用栅栏，等待上一帧绘制完成后，再开始绘制新的一帧，这个函数将会一直执行，只有当收到栅栏信号之后才会返回
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // 2、从交换链获取图像
        // 由于交换链是一个扩展功能，所以需要调用扩展的函数，其格式为：vk*KHR，它将获得交换链图像的索引
        uint32_t imageIndex;// 交换链图像索引
        // 检查此时的交换链是否满足渲染要求，如果不满足，则重建交换链
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {// 如果交换链处于次优SUBOPTIMAL状态，我们还是允许此交换链继续渲染
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // 等待上一帧完成之后，需要手动将栅栏信号重置为未发出信号的状态（如果重建了交换链，栅栏又会初始化成发出信号状态，所以需要在重建之后再重置栅栏，以避免死锁）
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // 3、记录命令缓冲
        vkResetCommandBuffer(commandBuffers[currentFrame],  0);// 先重置命令缓冲，以确保它能够被记录
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);// 根据命令缓冲和交换链图像记录命令缓冲

        // 4、提交命令缓冲
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};// 指定等待信号量
        // 指定在管道的哪个阶段进行等待，我们希望等待将颜色写入图像，直到它可用，所以我们指定写入颜色附件的图形管道的阶段
        // 这也就意味着，在图像尚不可用的阶段的操作还是可以先执行的，例如执行顶点着色器
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        // 指定要实际提交执行的命令缓冲
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        // 指定命令缓冲区完成执行后要发出信号的信号量，这里我们发出renderFinishedSemaphore信号量
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        // 提交命令缓冲，最后一个参数引用了围栏，当命令缓冲完成执行时，将发出此围栏信号
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // 5、呈现交换链图像
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        // 指定需要等待的信号量renderFinishedSemaphore，此信号量没有启用之前无法执行此步
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        // 指定图像所呈现给的交换链以及交换链图像索引
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // Optional

        // 提交向交换链呈现图像的请求，并检查此时的交换链是否满足渲染要求或者处于次优SUBOPTIMAL状态，如果不满足或者处于次优状态，则重建交换链
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;// 虽然许多驱动遇到窗口大小改变会返回VK_ERROR_OUT_OF_DATE_KHR，但是以防万一还是专门设定一个标志表示窗口大小是否改变
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        // 绘制完当前帧后，将索引指向下一帧
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // 交换链重建（每当窗口发生变化，就需要重建交换链，最典型的事件就是窗口大小发生变化）
    void recreateSwapChain() {
        // 处理一种特殊情况：窗口最小化，此时我们需要将此函数挂起，直到不再最小化
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);// 等待当前还在运行的操作处理完成

        // 重建新的交换链之前，需要把旧的交换链清除
        cleanupSwapChain();

        createSwapChain();// 重建交换链
        createImageViews();// 重建交换链图像视图
        createFramebuffers();// 重建交换链帧缓冲
    }

    // 删除当前交换链
    void cleanupSwapChain() {
        // 解除逻辑设备与帧缓冲的联系,删除所有图像视图
        for (auto & swapChainFramebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, swapChainFramebuffer, nullptr);
        }
        // 解除逻辑设备与图像视图的联系,删除所有图像视图
        for (auto & swapChainImageView : swapChainImageViews) {
            vkDestroyImageView(device, swapChainImageView, nullptr);
        }
        // 解除逻辑设备与交换链的联系，删除交换链
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
