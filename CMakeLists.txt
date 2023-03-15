cmake_minimum_required(VERSION 3.21)
project(VulkanTest)

set(CMAKE_CXX_STANDARD 17)

# glfw/glm/vulkan路径
set(GLFW_PATH "/opt/homebrew/Cellar/glfw/3.3.8")
set(GLM_PATH "/opt/homebrew/Cellar/glm/0.9.9.8")
set(VULKAN_PATH "/Users/halforange/Documents/library/VulkanSDK/macOS")

set(GLFW_INCLUDE_DIR ${GLFW_PATH}/include)
set(GLFW_LIBRARY ${GLFW_PATH}/lib/libglfw.3.3.dylib)
set(GLM_INCLUDE_DIR ${GLM_PATH}/include)
set(VULKAN_INCLUDE_DIR ${VULKAN_PATH}/include)

file (GLOB VULKAN_LIB_FILE
        ${VULKAN_PATH}/lib/libvulkan.1.3.239.dylib
        ${VULKAN_PATH}/lib/libvulkan.1.dylib
        )

file (GLOB MAIN_FILE
        main.cpp
        )

add_executable(${PROJECT_NAME} ${SRCFILES}
        ${MAIN_FILE})

include_directories(${GLFW_INCLUDE_DIR} ${GLM_INCLUDE_DIR} ${VULKAN_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}
        ${GLFW_LIBRARY}
        ${VULKAN_LIB_FILE}
        )
