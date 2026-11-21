// Vulkan 1.3 backend for the RHI: dynamic rendering + synchronization2, volk-loaded entry points.
// Simplifications to revisit with profiling (ADR 0008): one allocation per resource, host-visible
// buffers, destruction via wait-idle.

#include "nova/rhi/rhi.hpp"

#include "nova/core/assert.hpp"
#include "nova/core/log.hpp"
#include "nova/platform/vulkan.hpp"
#include "nova/platform/window.hpp"

#include <volk.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <format>
#include <optional>
#include <string_view>

namespace nova::rhi {
namespace {

constexpr u32 kFramesInFlight = 2;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkFormat kOffscreenFormat = VK_FORMAT_R8G8B8A8_SRGB;
constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

bool vk_ok(VkResult r, const char* what) {
    if (r == VK_SUCCESS) {
        return true;
    }
    NOVA_ERROR("RHI", "{} failed (VkResult {})", what, static_cast<int>(r));
    return false;
}

std::string version_string(u32 v) {
    return std::format("{}.{}.{}", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
}

// Dense slot storage behind typed handles (id = index + 1).
template <typename T>
class SlotPool {
public:
    u32 add(T value) {
        if (!free_.empty()) {
            const u32 i = free_.back();
            free_.pop_back();
            slots_[i] = std::move(value);
            return i + 1;
        }
        slots_.push_back(std::move(value));
        return static_cast<u32>(slots_.size());
    }
    T* get(u32 id) {
        return (id == 0 || id > slots_.size() || !slots_[id - 1]) ? nullptr : &*slots_[id - 1];
    }
    std::optional<T> take(u32 id) {
        T* p = get(id);
        if (!p) {
            return std::nullopt;
        }
        std::optional<T> out = std::move(slots_[id - 1]);
        slots_[id - 1].reset();
        free_.push_back(id - 1);
        return out;
    }
    template <typename F>
    void for_each(F&& f) {
        for (auto& s : slots_) {
            if (s) {
                f(*s);
            }
        }
    }

private:
    std::vector<std::optional<T>> slots_;
    std::vector<u32> free_;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    usize size = 0;
    VkDeviceAddress address = 0; // ray-tracing devices only
};

// Acceleration structure with its own storage; TLASes also own scratch + a host-visible instance array.
struct Accel {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    Buffer storage;
    Buffer scratch;
    Buffer instances;
    u32 capacity = 0;
};

struct Shader {
    VkShaderModule module = VK_NULL_HANDLE;
    ShaderStage stage{};
    std::string entry;
};

struct Pipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    std::vector<BindingDesc> bindings;
    u32 push_constant_size = 0;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct Texture {
    Image image;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    VkExtent2D extent{};
    bool sampled = false;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct Frame {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore image_available = VK_NULL_HANDLE;
    VkDescriptorPool descriptors = VK_NULL_HANDLE; // reset every frame
    bool timestamps_pending = false;
};

VkFormat to_vk(Format f) {
    switch (f) {
    case Format::RGBA8_sRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case Format::RGBA8_Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::RGBA16_Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Format::D32_Float: return VK_FORMAT_D32_SFLOAT;
    case Format::Undefined:
    case Format::BackBuffer: break;
    }
    return VK_FORMAT_UNDEFINED;
}

bool is_depth(VkFormat f) { return f == VK_FORMAT_D32_SFLOAT; }

VkImageMemoryBarrier2 image_barrier(VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                                    VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                    VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    return b;
}

void cmd_barriers(VkCommandBuffer cmd, std::span<const VkImageMemoryBarrier2> barriers) {
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = static_cast<u32>(barriers.size());
    dep.pImageMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);
}

class VulkanDevice;

class VulkanCommandList final : public CommandList {
public:
    explicit VulkanCommandList(VulkanDevice& device) : device_(device) {}
    using CommandList::begin_render_pass;
    void begin_render_pass(const RenderPassDesc& desc) override;
    void end_render_pass() override;
    void bind_pipeline(PipelineHandle pipeline) override;
    void bind_resources(std::span<const ResourceBinding> resources) override;
    void bind_vertex_buffer(BufferHandle buffer, usize offset) override;
    void bind_index_buffer(BufferHandle buffer, usize offset) override;
    void push_constants(const void* data, u32 size) override;
    void draw(u32 vertex_count, u32 first_vertex) override;
    void draw_indexed(u32 index_count, u32 first_index, i32 vertex_offset) override;
    void build_tlas(TlasHandle tlas, std::span<const TlasInstance> instances) override;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    Pipeline* bound = nullptr;
    bool in_pass = false;
    std::array<Texture*, 2> pass_textures{}; // render-target textures of the open pass

private:
    VulkanDevice& device_;
};

class VulkanDevice final : public Device {
public:
    explicit VulkanDevice(const DeviceDesc& desc) : desc_(desc), cmd_list_(*this) {}
    ~VulkanDevice() override;

    bool init();

    const DeviceInfo& info() const noexcept override { return info_; }
    Extent back_buffer_extent() const noexcept override { return {extent_.width, extent_.height}; }

    BufferHandle create_buffer(const BufferDesc& desc) override;
    void update_buffer(BufferHandle handle, const void* data, usize size, usize offset) override;
    ShaderHandle create_shader(const ShaderDesc& desc) override;
    PipelineHandle create_graphics_pipeline(const GraphicsPipelineDesc& desc) override;
    TextureHandle create_texture(const TextureDesc& desc) override;
    void destroy(BufferHandle handle) override;
    void destroy(ShaderHandle handle) override;
    void destroy(PipelineHandle handle) override;
    void destroy(TextureHandle handle) override;
    BlasHandle create_blas(const BlasDesc& desc) override;
    TlasHandle create_tlas(u32 max_instances) override;
    u64 buffer_address(BufferHandle handle) override;
    void destroy(BlasHandle handle) override;
    void destroy(TlasHandle handle) override;
    u32 frames_in_flight() const noexcept override { return kFramesInFlight; }

    CommandList* begin_frame() override;
    void end_frame() override;
    void resize(u32 width, u32 height) override;
    void wait_idle() override {
        if (device_) {
            vkDeviceWaitIdle(device_);
        }
    }

    f64 last_gpu_frame_ms() const noexcept override { return gpu_ms_; }
    u32 validation_error_count() const noexcept override { return validation_errors_.load(); }
    bool read_back(std::vector<u8>& rgba) override;

    // Used by VulkanCommandList.
    Buffer* buffer(BufferHandle h) { return buffers_.get(h.id); }
    Pipeline* pipeline(PipelineHandle h) { return pipelines_.get(h.id); }
    Texture* texture(TextureHandle h) { return textures_.get(h.id); }
    Accel* tlas(TlasHandle h) { return tlases_.get(h.id); }
    Accel* blas(BlasHandle h) { return blases_.get(h.id); }
    VkDevice vk_device() const { return device_; }
    VkDeviceSize scratch_alignment() const { return scratch_alignment_; }
    VkSampler sampler(Filter f) const {
        return f == Filter::Linear ? linear_sampler_ : (f == Filter::Nearest ? nearest_sampler_ : material_sampler_);
    }
    VkImage back_buffer_image() const { return swapchain_ ? swap_images_[image_index_] : offscreen_.image; }
    VkImageView back_buffer_view() const { return swapchain_ ? swap_views_[image_index_] : offscreen_.view; }
    const Image& depth() const { return depth_; }
    VkExtent2D extent() const { return extent_; }
    VkImageLayout back_buffer_layout = VK_IMAGE_LAYOUT_UNDEFINED;

private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                         VkDebugUtilsMessageTypeFlagsEXT,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                         void* user);

    bool create_instance();
    bool pick_physical_device();
    bool create_device();
    bool create_frames();
    bool create_swapchain();
    bool create_offscreen_target();
    bool create_depth();
    void destroy_targets();
    std::optional<u32> find_memory_type(u32 type_bits, VkMemoryPropertyFlags props) const;
    bool upload_texture(Texture& t, const void* rgba, u32 mip_levels);
    bool allocate_image(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkExtent2D extent,
                        Image& out, u32 mip_levels = 1);
    void destroy_image(Image& img);
    bool create_raw_buffer(usize size, VkBufferUsageFlags usage, Buffer& out, bool host_visible = true);
    void destroy_raw_buffer(Buffer& b);
    bool ray_tracing_supported() const;
    void destroy_accel(Accel& a);
    template <typename F>
    bool one_shot(const char* what, F&& record);

    DeviceDesc desc_;
    DeviceInfo info_;
    VulkanCommandList cmd_list_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    u32 queue_family_ = 0;
    f64 timestamp_period_ns_ = 0;
    u64 timestamp_mask_ = 0;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;

    std::array<Frame, kFramesInFlight> frames_{};
    u32 frame_index_ = 0;
    bool frame_active_ = false;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat color_format_ = kOffscreenFormat;
    std::vector<VkImage> swap_images_;
    std::vector<VkImageView> swap_views_;
    std::vector<VkSemaphore> render_finished_; // one per swapchain image
    u32 image_index_ = 0;
    bool swapchain_dirty_ = false;
    Image offscreen_;
    Image depth_;
    VkExtent2D extent_{};

    SlotPool<Buffer> buffers_;
    SlotPool<Shader> shaders_;
    SlotPool<Pipeline> pipelines_;
    SlotPool<Texture> textures_;
    SlotPool<Accel> blases_;
    SlotPool<Accel> tlases_;
    VkDeviceSize scratch_alignment_ = 256;
    VkSampler linear_sampler_ = VK_NULL_HANDLE;
    VkSampler nearest_sampler_ = VK_NULL_HANDLE;
    VkSampler material_sampler_ = VK_NULL_HANDLE;
    f32 max_anisotropy_ = 0.0f; // 0 = feature unavailable

    f64 gpu_ms_ = -1.0;
    std::atomic<u32> validation_errors_{0};
};

// ---------------------------------------------------------------- init / teardown

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDevice::debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                            VkDebugUtilsMessageTypeFlagsEXT,
                                                            const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                            void* user) {
    auto* self = static_cast<VulkanDevice*>(user);
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        self->validation_errors_.fetch_add(1);
        NOVA_ERROR("Vulkan", "{}", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        NOVA_WARN("Vulkan", "{}", data->pMessage);
    }
    return VK_FALSE;
}

bool VulkanDevice::init() {
    if (volkInitialize() != VK_SUCCESS) {
        NOVA_ERROR("RHI", "Vulkan loader not found");
        return false;
    }
    if (volkGetInstanceVersion() < VK_API_VERSION_1_3) {
        NOVA_ERROR("RHI", "Vulkan 1.3 instance required (found {})", version_string(volkGetInstanceVersion()));
        return false;
    }
    if (!create_instance() || !pick_physical_device() || !create_device() || !create_frames()) {
        return false;
    }
    if (!(surface_ ? create_swapchain() : (create_offscreen_target() && create_depth()))) {
        return false;
    }
    NOVA_INFO("RHI", "Vulkan {} on {}{} | {}x{} | validation {} | gpu timing {} | ray tracing {}", info_.api_version,
              info_.device_name, info_.software ? " (software)" : "", extent_.width, extent_.height,
              info_.validation ? "on" : "off", info_.gpu_timing ? "on" : "off", info_.ray_tracing ? "on" : "off");
    return true;
}

bool VulkanDevice::create_instance() {
    std::vector<const char*> extensions;
    if (desc_.window) {
        extensions = platform::vulkan_instance_extensions(*desc_.window);
        if (extensions.empty()) {
            NOVA_ERROR("RHI", "window cannot present with Vulkan");
            return false;
        }
    }

    std::vector<const char*> layers;
    if (desc_.validation) {
        u32 count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> props(count);
        vkEnumerateInstanceLayerProperties(&count, props.data());
        const bool found = std::ranges::any_of(
            props, [](const VkLayerProperties& p) { return std::strcmp(p.layerName, kValidationLayer) == 0; });
        if (found) {
            layers.push_back(kValidationLayer);
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            info_.validation = true;
        } else {
            NOVA_WARN("RHI", "validation requested but {} is not installed", kValidationLayer);
        }
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "NOVA";
    app.pEngineName = "NOVA Engine";
    app.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkDebugUtilsMessengerCreateInfoEXT dbg{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbg.pfnUserCallback = &debug_callback;
    dbg.pUserData = this;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pNext = info_.validation ? &dbg : nullptr; // also covers instance creation/destruction
    ci.pApplicationInfo = &app;
    ci.enabledLayerCount = static_cast<u32>(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    ci.enabledExtensionCount = static_cast<u32>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    if (!vk_ok(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance")) {
        return false;
    }
    volkLoadInstanceOnly(instance_);

    if (info_.validation &&
        !vk_ok(vkCreateDebugUtilsMessengerEXT(instance_, &dbg, nullptr, &messenger_), "debug messenger")) {
        return false;
    }
    if (desc_.window) {
        surface_ = reinterpret_cast<VkSurfaceKHR>(platform::create_vulkan_surface(*desc_.window, instance_));
        if (!surface_) {
            return false;
        }
    }
    return true;
}

bool VulkanDevice::pick_physical_device() {
    u32 count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    int best_score = -1;
    for (VkPhysicalDevice pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) {
            continue;
        }
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f13};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
        vkGetPhysicalDeviceFeatures2(pd, &f2);
        // shaderDrawParameters: SV_VertexID in Slang-generated SPIR-V (fullscreen passes).
        if (!f13.dynamicRendering || !f13.synchronization2 || !f11.shaderDrawParameters) {
            continue;
        }
        if (surface_) {
            u32 ext_count = 0;
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);
            std::vector<VkExtensionProperties> exts(ext_count);
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, exts.data());
            if (std::ranges::none_of(exts, [](const VkExtensionProperties& e) {
                    return std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
                })) {
                continue;
            }
        }

        u32 qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());
        std::optional<u32> family;
        for (u32 i = 0; i < qcount; ++i) {
            VkBool32 present = VK_TRUE;
            if (surface_) {
                vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface_, &present);
            }
            if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                family = i;
                break;
            }
        }
        if (!family) {
            continue;
        }

        int score = 0;
        switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = 3; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 2; break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score = 1; break;
        default: score = 0; break;
        }
        // Discrete NVIDIA first on multi-GPU (Optimus) machines; NOVA_GPU=<name substring> overrides.
        score = score * 10 + (props.vendorID == 0x10DE ? 1 : 0);
        if (const char* wanted = std::getenv("NOVA_GPU");
            wanted && *wanted && std::string_view(props.deviceName).find(wanted) != std::string_view::npos) {
            score += 1000;
        }
        if (score > best_score) {
            best_score = score;
            physical_ = pd;
            queue_family_ = *family;
            info_.device_name = props.deviceName;
            info_.api_version = version_string(props.apiVersion);
            info_.software = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
            timestamp_period_ns_ = props.limits.timestampPeriod;
            max_anisotropy_ = f2.features.samplerAnisotropy ? std::min(16.0f, props.limits.maxSamplerAnisotropy) : 0.0f;
            const u32 bits = qprops[*family].timestampValidBits;
            timestamp_mask_ = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
            info_.gpu_timing = bits > 0 && props.limits.timestampPeriod > 0;
        }
    }
    if (!physical_) {
        NOVA_ERROR("RHI", "no GPU with Vulkan 1.3, dynamic rendering and synchronization2{}",
                   surface_ ? " that can present to the window" : "");
        return false;
    }
    info_.backend = "Vulkan";
    vkGetPhysicalDeviceMemoryProperties(physical_, &mem_props_);
    return true;
}

bool VulkanDevice::ray_tracing_supported() const {
    if (!desc_.ray_tracing) {
        return false;
    }
    if (const char* off = std::getenv("NOVA_NO_RT"); off && *off && *off != '0') {
        return false;
    }
    u32 ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ext_count, exts.data());
    for (const char* needed : {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                               VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME}) {
        if (std::ranges::none_of(exts, [&](const VkExtensionProperties& e) {
                return std::strcmp(e.extensionName, needed) == 0;
            })) {
            return false;
        }
    }
    VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rq};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &as};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    vkGetPhysicalDeviceFeatures2(physical_, &f2);
    return as.accelerationStructure && rq.rayQuery && f12.bufferDeviceAddress;
}

bool VulkanDevice::create_device() {
    const f32 priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const bool rt = ray_tracing_supported();
    VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rq.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rq};
    as.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, rt ? &as : nullptr};
    f12.bufferDeviceAddress = rt ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f13};
    f11.shaderDrawParameters = VK_TRUE;

    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    f2.features.samplerAnisotropy = max_anisotropy_ > 0.0f ? VK_TRUE : VK_FALSE;

    std::vector<const char*> extensions;
    if (surface_) {
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    if (rt) {
        extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    ci.enabledExtensionCount = static_cast<u32>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    if (!vk_ok(vkCreateDevice(physical_, &ci, nullptr, &device_), "vkCreateDevice")) {
        return false;
    }
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    info_.ray_tracing = rt;
    if (rt) {
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &asp};
        vkGetPhysicalDeviceProperties2(physical_, &p2);
        scratch_alignment_ = std::max<VkDeviceSize>(asp.minAccelerationStructureScratchOffsetAlignment, 16);
    }
    return true;
}

bool VulkanDevice::create_frames() {
    for (Frame& f : frames_) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = queue_family_;
        if (!vk_ok(vkCreateCommandPool(device_, &pci, nullptr, &f.pool), "vkCreateCommandPool")) {
            return false;
        }
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (!vk_ok(vkAllocateCommandBuffers(device_, &ai, &f.cmd), "vkAllocateCommandBuffers") ||
            !vk_ok(vkCreateFence(device_, &fci, nullptr, &f.fence), "vkCreateFence") ||
            !vk_ok(vkCreateSemaphore(device_, &sci, nullptr, &f.image_available), "vkCreateSemaphore")) {
            return false;
        }
        std::vector<VkDescriptorPoolSize> sizes{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32768}, // per-draw material sets
        };
        if (info_.ray_tracing) {
            sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024});
            sizes.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1024});
        }
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 4096;
        dpci.poolSizeCount = static_cast<u32>(sizes.size());
        dpci.pPoolSizes = sizes.data();
        if (!vk_ok(vkCreateDescriptorPool(device_, &dpci, nullptr, &f.descriptors), "vkCreateDescriptorPool")) {
            return false;
        }
    }
    for (Filter filter : {Filter::Linear, Filter::Nearest}) {
        const VkFilter vf = filter == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = vf;
        sci.minFilter = vf;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        VkSampler& out = filter == Filter::Linear ? linear_sampler_ : nearest_sampler_;
        if (!vk_ok(vkCreateSampler(device_, &sci, nullptr, &out), "vkCreateSampler")) {
            return false;
        }
    }
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.anisotropyEnable = max_anisotropy_ > 0.0f ? VK_TRUE : VK_FALSE;
        sci.maxAnisotropy = std::max(1.0f, max_anisotropy_);
        sci.maxLod = VK_LOD_CLAMP_NONE;
        if (!vk_ok(vkCreateSampler(device_, &sci, nullptr, &material_sampler_), "vkCreateSampler")) {
            return false;
        }
    }
    if (info_.gpu_timing) {
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = 2 * kFramesInFlight;
        if (!vk_ok(vkCreateQueryPool(device_, &qci, nullptr, &query_pool_), "vkCreateQueryPool")) {
            info_.gpu_timing = false;
        }
    }
    return true;
}

std::optional<u32> VulkanDevice::find_memory_type(u32 type_bits, VkMemoryPropertyFlags props) const {
    for (u32 i = 0; i < mem_props_.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mem_props_.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return std::nullopt;
}

bool VulkanDevice::allocate_image(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                  VkExtent2D extent, Image& out, u32 mip_levels) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {extent.width, extent.height, 1};
    ici.mipLevels = mip_levels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!vk_ok(vkCreateImage(device_, &ici, nullptr, &out.image), "vkCreateImage")) {
        return false;
    }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, out.image, &req);
    const auto type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!type) {
        NOVA_ERROR("RHI", "no device-local memory type for image");
        return false;
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = *type;
    if (!vk_ok(vkAllocateMemory(device_, &mai, nullptr, &out.memory), "vkAllocateMemory") ||
        !vk_ok(vkBindImageMemory(device_, out.image, out.memory, 0), "vkBindImageMemory")) {
        return false;
    }
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = out.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspect, 0, mip_levels, 0, 1};
    return vk_ok(vkCreateImageView(device_, &vci, nullptr, &out.view), "vkCreateImageView");
}

void VulkanDevice::destroy_image(Image& img) {
    if (img.view) vkDestroyImageView(device_, img.view, nullptr);
    if (img.image) vkDestroyImage(device_, img.image, nullptr);
    if (img.memory) vkFreeMemory(device_, img.memory, nullptr);
    img = {};
}

bool VulkanDevice::create_depth() {
    return allocate_image(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, extent_,
                          depth_);
}

bool VulkanDevice::create_offscreen_target() {
    extent_ = {desc_.width, desc_.height};
    color_format_ = kOffscreenFormat;
    return allocate_image(kOffscreenFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          VK_IMAGE_ASPECT_COLOR_BIT, extent_, offscreen_);
}

bool VulkanDevice::create_swapchain() {
    vkDeviceWaitIdle(device_);
    VkSurfaceCapabilitiesKHR caps;
    if (!vk_ok(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps), "surface capabilities")) {
        return false;
    }
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFFu) {
        extent.width = std::clamp(desc_.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(desc_.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0) {
        swapchain_dirty_ = true; // minimized; retry next frame
        return true;
    }

    u32 count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &count, formats.data());
    VkSurfaceFormatKHR format = formats.at(0);
    for (const auto& f : formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format = f;
            break;
        }
    }

    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &count, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR; // always supported
    if (!desc_.vsync) {
        for (VkPresentModeKHR want : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
            if (std::ranges::find(modes, want) != modes.end()) {
                mode = want;
                break;
            }
        }
    }

    u32 image_count = caps.minImageCount + 1;
    if (caps.maxImageCount != 0) {
        image_count = std::min(image_count, caps.maxImageCount);
    }

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = image_count;
    ci.imageFormat = format.format;
    ci.imageColorSpace = format.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain_;

    VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
    if (!vk_ok(vkCreateSwapchainKHR(device_, &ci, nullptr, &new_swapchain), "vkCreateSwapchainKHR")) {
        return false;
    }
    destroy_targets();
    if (swapchain_) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    }
    swapchain_ = new_swapchain;
    extent_ = extent;
    color_format_ = format.format;

    vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr);
    swap_images_.resize(count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &count, swap_images_.data());
    for (VkImage img : swap_images_) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = color_format_;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        VkSemaphore sem = VK_NULL_HANDLE;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (!vk_ok(vkCreateImageView(device_, &vci, nullptr, &view), "swapchain view") ||
            !vk_ok(vkCreateSemaphore(device_, &sci, nullptr, &sem), "vkCreateSemaphore")) {
            return false;
        }
        swap_views_.push_back(view);
        render_finished_.push_back(sem);
    }
    swapchain_dirty_ = false;
    return create_depth();
}

void VulkanDevice::destroy_targets() {
    for (VkImageView v : swap_views_) vkDestroyImageView(device_, v, nullptr);
    for (VkSemaphore s : render_finished_) vkDestroySemaphore(device_, s, nullptr);
    swap_views_.clear();
    render_finished_.clear();
    swap_images_.clear();
    destroy_image(depth_);
}

VulkanDevice::~VulkanDevice() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        buffers_.for_each([&](Buffer& b) { destroy_raw_buffer(b); });
        shaders_.for_each([&](Shader& s) { vkDestroyShaderModule(device_, s.module, nullptr); });
        pipelines_.for_each([&](Pipeline& p) {
            vkDestroyPipeline(device_, p.pipeline, nullptr);
            vkDestroyPipelineLayout(device_, p.layout, nullptr);
            if (p.set_layout) vkDestroyDescriptorSetLayout(device_, p.set_layout, nullptr);
        });
        textures_.for_each([&](Texture& t) { destroy_image(t.image); });
        blases_.for_each([&](Accel& a) { destroy_accel(a); });
        tlases_.for_each([&](Accel& a) { destroy_accel(a); });
        if (linear_sampler_) vkDestroySampler(device_, linear_sampler_, nullptr);
        if (nearest_sampler_) vkDestroySampler(device_, nearest_sampler_, nullptr);
        if (material_sampler_) vkDestroySampler(device_, material_sampler_, nullptr);
        destroy_targets();
        destroy_image(offscreen_);
        if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        if (query_pool_) vkDestroyQueryPool(device_, query_pool_, nullptr);
        for (Frame& f : frames_) {
            if (f.descriptors) vkDestroyDescriptorPool(device_, f.descriptors, nullptr);
            if (f.image_available) vkDestroySemaphore(device_, f.image_available, nullptr);
            if (f.fence) vkDestroyFence(device_, f.fence, nullptr);
            if (f.pool) vkDestroyCommandPool(device_, f.pool, nullptr);
        }
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) {
        if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (messenger_) vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
        vkDestroyInstance(instance_, nullptr);
    }
}

// ---------------------------------------------------------------- resources

bool VulkanDevice::create_raw_buffer(usize size, VkBufferUsageFlags usage, Buffer& out, bool host_visible) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!vk_ok(vkCreateBuffer(device_, &bci, nullptr, &out.buffer), "vkCreateBuffer")) {
        return false;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, out.buffer, &req);
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags wanted = host_visible ? host : 0;
    auto type = find_memory_type(req.memoryTypeBits, wanted | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); // ReBAR/UMA
    if (!type) {
        type = find_memory_type(req.memoryTypeBits, wanted);
    }
    if (!type) {
        NOVA_ERROR("RHI", "no suitable memory type for buffer");
        vkDestroyBuffer(device_, out.buffer, nullptr);
        out = {};
        return false;
    }
    const bool device_address = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, device_address ? &flags : nullptr};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = *type;
    VkResult alloc_res = vkAllocateMemory(device_, &mai, nullptr, &out.memory);
    if (alloc_res != VK_SUCCESS) {
        auto fallback_type = find_memory_type(req.memoryTypeBits, wanted);
        if (fallback_type && *fallback_type != *type) {
            mai.memoryTypeIndex = *fallback_type;
            alloc_res = vkAllocateMemory(device_, &mai, nullptr, &out.memory);
        }
    }
    if (!vk_ok(alloc_res, "vkAllocateMemory") ||
        !vk_ok(vkBindBufferMemory(device_, out.buffer, out.memory, 0), "vkBindBufferMemory") ||
        (host_visible && !vk_ok(vkMapMemory(device_, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped), "vkMapMemory"))) {
        destroy_raw_buffer(out);
        return false;
    }
    out.size = size;
    if (device_address) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = out.buffer;
        out.address = vkGetBufferDeviceAddress(device_, &ai);
    }
    return true;
}

void VulkanDevice::destroy_raw_buffer(Buffer& b) {
    if (b.buffer) vkDestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device_, b.memory, nullptr); // implicitly unmaps
    b = {};
}

BufferHandle VulkanDevice::create_buffer(const BufferDesc& desc) {
    NOVA_ASSERT(desc.size > 0, "buffer size must be non-zero");
    VkBufferUsageFlags usage = 0;
    switch (desc.usage) {
    case BufferUsage::Vertex: usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; break;
    case BufferUsage::Index: usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT; break;
    case BufferUsage::Uniform: usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; break;
    case BufferUsage::Storage: usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; break;
    }
    if (info_.ray_tracing && desc.usage != BufferUsage::Uniform) {
        // Geometry doubles as BLAS build input and is read through shader pointers by hit shading.
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }
    Buffer b;
    if (!create_raw_buffer(desc.size, usage, b)) {
        return {};
    }
    if (desc.initial_data) {
        std::memcpy(b.mapped, desc.initial_data, desc.size);
    }
    return {buffers_.add(b)};
}

void VulkanDevice::update_buffer(BufferHandle handle, const void* data, usize size, usize offset) {
    Buffer* b = buffers_.get(handle.id);
    NOVA_VERIFY(b, "invalid buffer handle");
    NOVA_VERIFY(offset <= b->size && size <= b->size - offset, "buffer update out of range");
    std::memcpy(static_cast<std::byte*>(b->mapped) + offset, data, size);
}

ShaderHandle VulkanDevice::create_shader(const ShaderDesc& desc) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = desc.spirv.size_bytes();
    ci.pCode = desc.spirv.data();
    Shader s{VK_NULL_HANDLE, desc.stage, desc.entry_point};
    if (!vk_ok(vkCreateShaderModule(device_, &ci, nullptr, &s.module), "vkCreateShaderModule")) {
        return {};
    }
    return {shaders_.add(std::move(s))};
}

PipelineHandle VulkanDevice::create_graphics_pipeline(const GraphicsPipelineDesc& desc) {
    Shader* vs = shaders_.get(desc.vertex_shader.id);
    Shader* fs = shaders_.get(desc.fragment_shader.id);
    NOVA_VERIFY(vs && fs, "pipeline needs valid vertex and fragment shaders");
    NOVA_VERIFY(desc.push_constant_size <= 128, "push constants are limited to 128 bytes");

    Pipeline p;
    p.push_constant_size = desc.push_constant_size;
    p.bindings = desc.bindings;

    std::vector<VkDescriptorSetLayoutBinding> set_bindings;
    std::vector<VkSampler> immutable(desc.bindings.size());
    for (usize i = 0; i < desc.bindings.size(); ++i) {
        const BindingDesc& b = desc.bindings[i];
        VkDescriptorSetLayoutBinding lb{};
        lb.binding = b.binding;
        lb.descriptorCount = 1;
        lb.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        if (b.type == BindingType::UniformBuffer) {
            lb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        } else if (b.type == BindingType::StorageBuffer) {
            lb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        } else if (b.type == BindingType::AccelerationStructure) {
            NOVA_VERIFY(info_.ray_tracing, "acceleration structure binding needs a ray-tracing device");
            lb.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        } else {
            lb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            immutable[i] = sampler(b.filter);
            lb.pImmutableSamplers = &immutable[i];
        }
        set_bindings.push_back(lb);
    }
    if (!set_bindings.empty()) {
        VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        sli.bindingCount = static_cast<u32>(set_bindings.size());
        sli.pBindings = set_bindings.data();
        if (!vk_ok(vkCreateDescriptorSetLayout(device_, &sli, nullptr, &p.set_layout), "descriptor set layout")) {
            return {};
        }
    }

    VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, desc.push_constant_size};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = p.set_layout ? 1 : 0;
    lci.pSetLayouts = &p.set_layout;
    lci.pushConstantRangeCount = desc.push_constant_size ? 1 : 0;
    lci.pPushConstantRanges = &range;
    if (!vk_ok(vkCreatePipelineLayout(device_, &lci, nullptr, &p.layout), "vkCreatePipelineLayout")) {
        if (p.set_layout) vkDestroyDescriptorSetLayout(device_, p.set_layout, nullptr);
        return {};
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs->module;
    stages[0].pName = vs->entry.c_str();
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs->module;
    stages[1].pName = fs->entry.c_str();

    VkVertexInputBindingDescription binding{0, desc.vertex_stride, VK_VERTEX_INPUT_RATE_VERTEX};
    std::vector<VkVertexInputAttributeDescription> attrs;
    for (const auto& a : desc.attributes) {
        VkFormat f = VK_FORMAT_R32G32B32_SFLOAT;
        switch (a.format) {
        case VertexFormat::Float2: f = VK_FORMAT_R32G32_SFLOAT; break;
        case VertexFormat::Float3: f = VK_FORMAT_R32G32B32_SFLOAT; break;
        case VertexFormat::Float4: f = VK_FORMAT_R32G32B32A32_SFLOAT; break;
        }
        attrs.push_back({a.location, 0, f, a.offset});
    }
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = desc.vertex_stride ? 1 : 0;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = static_cast<u32>(attrs.size());
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1;
    vps.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = desc.cull == CullMode::Back ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    // CCW front faces in world space; the Y flip in math::perspective preserves on-screen winding.
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    if (desc.depth_bias) {
        rs.depthBiasEnable = VK_TRUE;
        rs.depthBiasConstantFactor = 1.5f;
        rs.depthBiasSlopeFactor = 2.0f;
    }
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = desc.depth_test;
    ds.depthWriteEnable = desc.depth_test && desc.depth_write;
    switch (desc.depth_compare) {
    case CompareOp::Less: ds.depthCompareOp = VK_COMPARE_OP_LESS; break;
    case CompareOp::LessOrEqual: ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; break;
    case CompareOp::Always: ds.depthCompareOp = VK_COMPARE_OP_ALWAYS; break;
    }
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                           VK_COLOR_COMPONENT_A_BIT;
    if (desc.alpha_blend) {
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    const VkFormat color_format =
        desc.color_format == Format::BackBuffer ? color_format_ : to_vk(desc.color_format);
    const bool has_color = color_format != VK_FORMAT_UNDEFINED;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = has_color ? 1 : 0;
    cb.pAttachments = &blend;
    const std::array<VkDynamicState, 2> dyn_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = static_cast<u32>(dyn_states.size());
    dyn.pDynamicStates = dyn_states.data();

    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = has_color ? 1 : 0;
    rendering.pColorAttachmentFormats = &color_format;
    rendering.depthAttachmentFormat = to_vk(desc.depth_format);

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
    ci.stageCount = static_cast<u32>(stages.size());
    ci.pStages = stages.data();
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vps;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &cb;
    ci.pDynamicState = &dyn;
    ci.layout = p.layout;
    if (!vk_ok(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline),
               "vkCreateGraphicsPipelines")) {
        vkDestroyPipelineLayout(device_, p.layout, nullptr);
        if (p.set_layout) vkDestroyDescriptorSetLayout(device_, p.set_layout, nullptr);
        return {};
    }
    return {pipelines_.add(std::move(p))};
}

TextureHandle VulkanDevice::create_texture(const TextureDesc& desc) {
    NOVA_VERIFY(desc.width > 0 && desc.height > 0, "texture size must be non-zero");
    Texture t;
    t.format = to_vk(desc.format);
    NOVA_VERIFY(t.format != VK_FORMAT_UNDEFINED, "texture needs a concrete format");
    t.aspect = is_depth(t.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    t.extent = {desc.width, desc.height};
    t.sampled = desc.sampled;
    VkImageUsageFlags usage = 0;
    if (desc.render_target) {
        usage |= is_depth(t.format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if (desc.sampled) {
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    u32 mip_levels = 1;
    if (desc.initial_data) {
        NOVA_VERIFY(!desc.render_target && (t.format == VK_FORMAT_R8G8B8A8_SRGB || t.format == VK_FORMAT_R8G8B8A8_UNORM),
                    "initial texture data needs a sampled-only RGBA8 texture");
        usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (desc.mipmaps) {
            mip_levels = static_cast<u32>(std::bit_width(std::max(desc.width, desc.height)));
        }
    }
    if (!allocate_image(t.format, usage, t.aspect, t.extent, t.image, mip_levels)) {
        destroy_image(t.image);
        return {};
    }
    if (desc.initial_data && !upload_texture(t, desc.initial_data, mip_levels)) {
        destroy_image(t.image);
        return {};
    }
    return {textures_.add(t)};
}

// Copies RGBA8 pixels into mip 0, blits the rest of the chain and leaves every level
// SHADER_READ_ONLY. Synchronous: meant for load time, not for the middle of a frame.
bool VulkanDevice::upload_texture(Texture& t, const void* rgba, u32 mip_levels) {
    NOVA_VERIFY(!frame_active_, "textures with initial data must be created outside begin_frame/end_frame");
    const usize bytes = static_cast<usize>(t.extent.width) * t.extent.height * 4;
    Buffer staging;
    if (!create_raw_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
        return false;
    }
    std::memcpy(staging.mapped, rgba, bytes);
    vkDeviceWaitIdle(device_);

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = frames_[frame_index_].pool; // idle after the wait above
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    auto barrier = [&](u32 level, u32 count, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 src_stage,
                       VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
        VkImageMemoryBarrier2 b = image_barrier(t.image.image, VK_IMAGE_ASPECT_COLOR_BIT, from, to, src_stage,
                                                src_access, dst_stage, dst_access);
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, count, 0, 1};
        cmd_barriers(cmd, {&b, 1});
    };
    constexpr auto kTransfer = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier(0, mip_levels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE, kTransfer, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {t.extent.width, t.extent.height, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, t.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    i32 w = static_cast<i32>(t.extent.width);
    i32 h = static_cast<i32>(t.extent.height);
    for (u32 level = 1; level < mip_levels; ++level) {
        barrier(level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, kTransfer,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, kTransfer, VK_ACCESS_2_TRANSFER_READ_BIT);
        const i32 nw = std::max(1, w / 2);
        const i32 nh = std::max(1, h / 2);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
        blit.srcOffsets[1] = {w, h, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
        blit.dstOffsets[1] = {nw, nh, 1};
        vkCmdBlitImage(cmd, t.image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.image.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        barrier(level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                kTransfer, VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        w = nw;
        h = nh;
    }
    barrier(mip_levels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            kTransfer, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_info.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;
    const bool ok = vk_ok(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE), "texture upload submit") &&
                    vk_ok(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device_, frames_[frame_index_].pool, 1, &cmd);
    destroy_raw_buffer(staging);
    t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return ok;
}

void VulkanDevice::destroy(TextureHandle handle) {
    if (auto t = textures_.take(handle.id)) {
        vkDeviceWaitIdle(device_);
        destroy_image(t->image);
    }
}

void VulkanDevice::destroy(BufferHandle handle) {
    if (auto b = buffers_.take(handle.id)) {
        vkDeviceWaitIdle(device_);
        destroy_raw_buffer(*b);
    }
}

void VulkanDevice::destroy(ShaderHandle handle) {
    if (auto s = shaders_.take(handle.id)) {
        vkDestroyShaderModule(device_, s->module, nullptr); // not referenced after pipeline creation
    }
}

void VulkanDevice::destroy(PipelineHandle handle) {
    if (auto p = pipelines_.take(handle.id)) {
        vkDeviceWaitIdle(device_);
        vkDestroyPipeline(device_, p->pipeline, nullptr);
        vkDestroyPipelineLayout(device_, p->layout, nullptr);
        if (p->set_layout) vkDestroyDescriptorSetLayout(device_, p->set_layout, nullptr);
    }
}

// ---------------------------------------------------------------- ray tracing

template <typename F>
bool VulkanDevice::one_shot(const char* what, F&& record) {
    NOVA_VERIFY(!frame_active_, "one-shot GPU work must run outside begin_frame/end_frame");
    vkDeviceWaitIdle(device_);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = frames_[frame_index_].pool; // idle after the wait above
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!vk_ok(vkAllocateCommandBuffers(device_, &ai, &cmd), what)) {
        return false;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    record(cmd);
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_info.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;
    const bool ok = vk_ok(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE), what) &&
                    vk_ok(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device_, frames_[frame_index_].pool, 1, &cmd);
    return ok;
}

void VulkanDevice::destroy_accel(Accel& a) {
    if (a.handle) vkDestroyAccelerationStructureKHR(device_, a.handle, nullptr);
    destroy_raw_buffer(a.storage);
    destroy_raw_buffer(a.scratch);
    destroy_raw_buffer(a.instances);
    a = {};
}

namespace {
constexpr VkBufferUsageFlags kAccelStorageUsage =
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
constexpr VkBufferUsageFlags kScratchUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

VkDeviceAddress align_up(VkDeviceAddress v, VkDeviceSize a) { return (v + a - 1) / a * a; }
} // namespace

BlasHandle VulkanDevice::create_blas(const BlasDesc& desc) {
    if (!info_.ray_tracing || desc.vertex_count == 0 || desc.index_count < 3) {
        return {};
    }
    Buffer* vb = buffers_.get(desc.vertices.id);
    Buffer* ib = buffers_.get(desc.indices.id);
    NOVA_VERIFY(vb && ib && vb->address && ib->address, "BLAS needs valid geometry buffers");

    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& tri = geom.geometry.triangles;
    tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.vertexData.deviceAddress = vb->address;
    tri.vertexStride = desc.vertex_stride;
    tri.maxVertex = desc.vertex_count - 1;
    tri.indexType = VK_INDEX_TYPE_UINT32;
    tri.indexData.deviceAddress = ib->address;

    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geom;
    const u32 prims = desc.index_count / 3;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &prims,
                                            &sizes);

    Accel a;
    Buffer scratch;
    if (!create_raw_buffer(sizes.accelerationStructureSize, kAccelStorageUsage, a.storage, false) ||
        !create_raw_buffer(sizes.buildScratchSize + scratch_alignment_, kScratchUsage, scratch, false)) {
        destroy_raw_buffer(scratch);
        destroy_accel(a);
        return {};
    }
    VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    ci.buffer = a.storage.buffer;
    ci.size = sizes.accelerationStructureSize;
    ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (!vk_ok(vkCreateAccelerationStructureKHR(device_, &ci, nullptr, &a.handle), "create BLAS")) {
        destroy_raw_buffer(scratch);
        destroy_accel(a);
        return {};
    }
    build.dstAccelerationStructure = a.handle;
    build.scratchData.deviceAddress = align_up(scratch.address, scratch_alignment_);
    VkAccelerationStructureBuildRangeInfoKHR range{prims, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
    const bool ok = one_shot("BLAS build", [&](VkCommandBuffer cmd) {
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &ranges);
    });
    destroy_raw_buffer(scratch);
    if (!ok) {
        destroy_accel(a);
        return {};
    }
    return {blases_.add(a)};
}

TlasHandle VulkanDevice::create_tlas(u32 max_instances) {
    if (!info_.ray_tracing) {
        return {};
    }
    max_instances = std::max(1u, max_instances);
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geom;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                                            &max_instances, &sizes);
    Accel a;
    a.capacity = max_instances;
    if (!create_raw_buffer(sizes.accelerationStructureSize, kAccelStorageUsage, a.storage, false) ||
        !create_raw_buffer(sizes.buildScratchSize + scratch_alignment_, kScratchUsage, a.scratch, false) ||
        !create_raw_buffer(sizeof(VkAccelerationStructureInstanceKHR) * max_instances,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           a.instances)) {
        destroy_accel(a);
        return {};
    }
    VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    ci.buffer = a.storage.buffer;
    ci.size = sizes.accelerationStructureSize;
    ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (!vk_ok(vkCreateAccelerationStructureKHR(device_, &ci, nullptr, &a.handle), "create TLAS")) {
        destroy_accel(a);
        return {};
    }
    return {tlases_.add(a)};
}

u64 VulkanDevice::buffer_address(BufferHandle handle) {
    const Buffer* b = buffers_.get(handle.id);
    return b ? b->address : 0;
}

void VulkanDevice::destroy(BlasHandle handle) {
    if (auto a = blases_.take(handle.id)) {
        vkDeviceWaitIdle(device_);
        destroy_accel(*a);
    }
}

void VulkanDevice::destroy(TlasHandle handle) {
    if (auto a = tlases_.take(handle.id)) {
        vkDeviceWaitIdle(device_);
        destroy_accel(*a);
    }
}

void VulkanCommandList::build_tlas(TlasHandle handle, std::span<const TlasInstance> instances) {
    NOVA_ASSERT(!in_pass, "build_tlas inside a render pass");
    Accel* a = device_.tlas(handle);
    NOVA_VERIFY(a && instances.size() <= a->capacity, "invalid TLAS or too many instances");
    auto* dst = static_cast<VkAccelerationStructureInstanceKHR*>(a->instances.mapped);
    u32 count = 0;
    for (const TlasInstance& in : instances) {
        Accel* blas = device_.blas(in.blas);
        if (!blas) {
            continue;
        }
        VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        ai.accelerationStructure = blas->handle;
        VkAccelerationStructureInstanceKHR& out = dst[count++];
        std::memcpy(&out.transform, in.transform, sizeof(out.transform));
        out.instanceCustomIndex = in.custom_index & 0xFFFFFFu;
        out.mask = in.mask;
        out.instanceShaderBindingTableRecordOffset = 0;
        out.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR | VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        out.accelerationStructureReference = vkGetAccelerationStructureDeviceAddressKHR(device_.vk_device(), &ai);
    }

    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.data.deviceAddress = a->instances.address;
    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.dstAccelerationStructure = a->handle;
    build.geometryCount = 1;
    build.pGeometries = &geom;
    build.scratchData.deviceAddress = align_up(a->scratch.address, device_.scratch_alignment());
    VkAccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;

    // Earlier traces / builds this frame (scratch reuse) must finish before rebuilding.
    VkMemoryBarrier2 pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    pre.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    pre.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    pre.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    pre.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &pre;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &ranges);

    VkMemoryBarrier2 post{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    post.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    post.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    post.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    post.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    dep.pMemoryBarriers = &post;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------- frames

void VulkanDevice::resize(u32 width, u32 height) {
    desc_.width = width;
    desc_.height = height;
    if (surface_) {
        swapchain_dirty_ = true;
    } else if (width != extent_.width || height != extent_.height) {
        vkDeviceWaitIdle(device_);
        destroy_image(offscreen_);
        destroy_image(depth_);
        back_buffer_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        NOVA_VERIFY(create_offscreen_target() && create_depth(), "offscreen target resize failed");
    }
}

CommandList* VulkanDevice::begin_frame() {
    NOVA_ASSERT(!frame_active_, "begin_frame called twice");
    if (swapchain_dirty_ && (!create_swapchain() || swapchain_dirty_)) {
        return nullptr; // still minimized or recreation failed
    }
    Frame& f = frames_[frame_index_];
    vkWaitForFences(device_, 1, &f.fence, VK_TRUE, UINT64_MAX);

    if (f.timestamps_pending) {
        std::array<u64, 2> ts{};
        if (vkGetQueryPoolResults(device_, query_pool_, frame_index_ * 2, 2, sizeof(ts), ts.data(), sizeof(u64),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const u64 delta = ((ts[1] & timestamp_mask_) - (ts[0] & timestamp_mask_)) & timestamp_mask_;
            gpu_ms_ = static_cast<f64>(delta) * timestamp_period_ns_ / 1e6;
        }
        f.timestamps_pending = false;
    }

    if (swapchain_) {
        const VkResult r =
            vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, f.image_available, VK_NULL_HANDLE, &image_index_);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain_dirty_ = true;
            return nullptr;
        }
        if (r == VK_SUBOPTIMAL_KHR) {
            swapchain_dirty_ = true; // still usable this frame
        } else if (!vk_ok(r, "vkAcquireNextImageKHR")) {
            return nullptr;
        }
    }

    vkResetFences(device_, 1, &f.fence);
    vkResetCommandPool(device_, f.pool, 0);
    vkResetDescriptorPool(device_, f.descriptors, 0);
    cmd_list_.descriptors = f.descriptors;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);
    if (info_.gpu_timing) {
        vkCmdResetQueryPool(f.cmd, query_pool_, frame_index_ * 2, 2);
        vkCmdWriteTimestamp2(f.cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, query_pool_, frame_index_ * 2);
    }

    back_buffer_layout = VK_IMAGE_LAYOUT_UNDEFINED; // contents are cleared each frame
    cmd_list_.cmd = f.cmd;
    cmd_list_.bound = nullptr;
    frame_active_ = true;
    return &cmd_list_;
}

void VulkanDevice::end_frame() {
    NOVA_ASSERT(frame_active_, "end_frame without begin_frame");
    NOVA_ASSERT(!cmd_list_.in_pass, "render pass still open at end_frame");
    Frame& f = frames_[frame_index_];

    const VkImageLayout final_layout = swapchain_ ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const auto barrier = image_barrier(
        back_buffer_image(), VK_IMAGE_ASPECT_COLOR_BIT, back_buffer_layout, final_layout,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        swapchain_ ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        swapchain_ ? VK_ACCESS_2_NONE : VK_ACCESS_2_TRANSFER_READ_BIT);
    cmd_barriers(f.cmd, {&barrier, 1});
    back_buffer_layout = final_layout;

    if (info_.gpu_timing) {
        vkCmdWriteTimestamp2(f.cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, query_pool_, frame_index_ * 2 + 1);
    }
    vkEndCommandBuffer(f.cmd);

    VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_info.commandBuffer = f.cmd;
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = f.image_available;
    wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = swapchain_ ? render_finished_[image_index_] : VK_NULL_HANDLE;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;
    if (swapchain_) {
        submit.waitSemaphoreInfoCount = 1;
        submit.pWaitSemaphoreInfos = &wait;
        submit.signalSemaphoreInfoCount = 1;
        submit.pSignalSemaphoreInfos = &signal;
    }
    vk_ok(vkQueueSubmit2(queue_, 1, &submit, f.fence), "vkQueueSubmit2");
    f.timestamps_pending = info_.gpu_timing;

    if (swapchain_) {
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &render_finished_[image_index_];
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &image_index_;
        const VkResult r = vkQueuePresentKHR(queue_, &present);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
            swapchain_dirty_ = true;
        } else {
            vk_ok(r, "vkQueuePresentKHR");
        }
    }
    frame_index_ = (frame_index_ + 1) % kFramesInFlight;
    frame_active_ = false;
}

bool VulkanDevice::read_back(std::vector<u8>& rgba) {
    if (swapchain_ || back_buffer_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        NOVA_ERROR("RHI", "read_back needs a headless device with at least one finished frame");
        return false;
    }
    vkDeviceWaitIdle(device_);
    const usize bytes = static_cast<usize>(extent_.width) * extent_.height * 4;
    Buffer staging;
    if (!create_raw_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, staging)) {
        return false;
    }

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = frames_[frame_index_].pool; // idle after wait above
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {extent_.width, extent_.height, 1};
    vkCmdCopyImageToBuffer(cmd, offscreen_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer, 1, &region);
    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_info.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmd_info;
    const bool ok = vk_ok(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE), "readback submit") &&
                    vk_ok(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device_, frames_[frame_index_].pool, 1, &cmd);
    if (ok) {
        rgba.resize(bytes);
        std::memcpy(rgba.data(), staging.mapped, bytes);
    }
    destroy_raw_buffer(staging);
    return ok;
}

// ---------------------------------------------------------------- command list

void VulkanCommandList::begin_render_pass(const RenderPassDesc& desc) {
    NOVA_ASSERT(!in_pass, "render pass already open");
    constexpr VkPipelineStageFlags2 kDepthStages =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    constexpr VkAccessFlags2 kDepthAccess =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    // Attachments are always cleared, so previous contents are discarded (old layout UNDEFINED); the
    // source stages still order this pass after earlier writes and shader reads of the same image.
    constexpr VkPipelineStageFlags2 kPrevUse = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | kDepthStages |
                                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    std::array<VkImageMemoryBarrier2, 2> barriers{};
    u32 barrier_count = 0;
    pass_textures = {};
    VkExtent2D extent{};

    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = desc.load_color ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    std::memcpy(color.clearValue.color.float32, desc.clear.color, sizeof(desc.clear.color));
    bool has_color = true;
    if (desc.back_buffer) {
        color.imageView = device_.back_buffer_view();
        const VkImageLayout old_layout = desc.load_color ? device_.back_buffer_layout : VK_IMAGE_LAYOUT_UNDEFINED;
        const VkAccessFlags2 src_access = desc.load_color ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_NONE;
        const VkAccessFlags2 dst_access = desc.load_color
            ? (VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
            : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[barrier_count++] = image_barrier(
            device_.back_buffer_image(), VK_IMAGE_ASPECT_COLOR_BIT, old_layout,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, kPrevUse, src_access,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, dst_access);
        device_.back_buffer_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        extent = device_.extent();
    } else if (desc.color.valid()) {
        Texture* t = device_.texture(desc.color);
        NOVA_VERIFY(t && !is_depth(t->format), "invalid colour render target");
        color.imageView = t->image.view;
        const VkImageLayout old_layout = desc.load_color ? t->layout : VK_IMAGE_LAYOUT_UNDEFINED;
        const VkAccessFlags2 src_access = desc.load_color ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_NONE;
        const VkAccessFlags2 dst_access = desc.load_color
            ? (VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
            : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[barrier_count++] = image_barrier(
            t->image.image, t->aspect, old_layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, kPrevUse,
            src_access, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, dst_access);
        t->layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        pass_textures[0] = t;
        extent = t->extent;
    } else {
        has_color = false;
    }

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.clearValue.depthStencil = {desc.clear.depth, 0};
    bool has_depth = true;
    if (desc.depth.valid()) {
        Texture* t = device_.texture(desc.depth);
        NOVA_VERIFY(t && is_depth(t->format), "invalid depth render target");
        depth.imageView = t->image.view;
        depth.storeOp = t->sampled ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        barriers[barrier_count++] = image_barrier(t->image.image, t->aspect, VK_IMAGE_LAYOUT_UNDEFINED,
                                                  VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, kPrevUse,
                                                  VK_ACCESS_2_NONE, kDepthStages, kDepthAccess);
        t->layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        pass_textures[1] = t;
        if (!has_color) {
            extent = t->extent;
        }
    } else if (desc.default_depth) {
        depth.imageView = device_.depth().view;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        barriers[barrier_count++] = image_barrier(device_.depth().image, VK_IMAGE_ASPECT_DEPTH_BIT,
                                                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                                  kDepthStages, kDepthAccess, kDepthStages, kDepthAccess);
        if (!has_color) {
            extent = device_.extent();
        }
    } else {
        has_depth = false;
    }
    NOVA_VERIFY(has_color || has_depth, "render pass needs at least one attachment");
    cmd_barriers(cmd, {barriers.data(), barrier_count});

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = has_color ? 1 : 0;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = has_depth ? &depth : nullptr;
    vkCmdBeginRendering(cmd, &ri);

    const VkViewport viewport{0, 0, static_cast<f32>(extent.width), static_cast<f32>(extent.height), 0, 1};
    const VkRect2D scissor{{0, 0}, extent};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    in_pass = true;
}

void VulkanCommandList::end_render_pass() {
    NOVA_ASSERT(in_pass, "no render pass open");
    vkCmdEndRendering(cmd);
    in_pass = false;

    // Make sampled render targets readable by later passes.
    std::array<VkImageMemoryBarrier2, 2> barriers{};
    u32 count = 0;
    for (Texture* t : pass_textures) {
        if (!t || !t->sampled) {
            continue;
        }
        const bool d = is_depth(t->format);
        barriers[count++] = image_barrier(
            t->image.image, t->aspect, t->layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            d ? VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            d ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        t->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if (count > 0) {
        cmd_barriers(cmd, {barriers.data(), count});
    }
    pass_textures = {};
}

void VulkanCommandList::bind_resources(std::span<const ResourceBinding> resources) {
    NOVA_VERIFY(bound && bound->set_layout, "bound pipeline has no resource bindings");
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descriptors;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &bound->set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    NOVA_VERIFY(vkAllocateDescriptorSets(device_.vk_device(), &ai, &set) == VK_SUCCESS,
                "per-frame descriptor pool exhausted");

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    std::vector<VkDescriptorImageInfo> image_infos;
    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> as_infos;
    std::vector<VkAccelerationStructureKHR> as_handles;
    buffer_infos.reserve(resources.size());
    image_infos.reserve(resources.size());
    as_infos.reserve(resources.size());
    as_handles.reserve(resources.size());
    for (const ResourceBinding& r : resources) {
        const auto it = std::ranges::find_if(bound->bindings, [&](const BindingDesc& b) { return b.binding == r.binding; });
        NOVA_VERIFY(it != bound->bindings.end(), "binding not declared by the pipeline");
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = r.binding;
        w.descriptorCount = 1;
        if (it->type == BindingType::UniformBuffer || it->type == BindingType::StorageBuffer) {
            Buffer* b = device_.buffer(r.buffer);
            NOVA_VERIFY(b, "invalid uniform/storage buffer");
            buffer_infos.push_back({b->buffer, 0, b->size});
            w.descriptorType = it->type == BindingType::UniformBuffer ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                      : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &buffer_infos.back();
        } else if (it->type == BindingType::AccelerationStructure) {
            Accel* a = device_.tlas(r.tlas);
            NOVA_VERIFY(a, "invalid TLAS handle");
            as_handles.push_back(a->handle);
            VkWriteDescriptorSetAccelerationStructureKHR asw{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            asw.accelerationStructureCount = 1;
            asw.pAccelerationStructures = &as_handles.back();
            as_infos.push_back(asw);
            w.pNext = &as_infos.back();
            w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        } else {
            Texture* t = device_.texture(r.texture);
            NOVA_VERIFY(t && t->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        "texture must be written by an earlier pass before it is sampled");
            image_infos.push_back({VK_NULL_HANDLE, t->image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &image_infos.back();
        }
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(device_.vk_device(), static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bound->layout, 0, 1, &set, 0, nullptr);
}

void VulkanCommandList::bind_pipeline(PipelineHandle handle) {
    bound = device_.pipeline(handle);
    NOVA_VERIFY(bound, "invalid pipeline handle");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bound->pipeline);
}

void VulkanCommandList::bind_vertex_buffer(BufferHandle handle, usize offset) {
    Buffer* b = device_.buffer(handle);
    NOVA_VERIFY(b, "invalid vertex buffer handle");
    const VkDeviceSize off = offset;
    vkCmdBindVertexBuffers(cmd, 0, 1, &b->buffer, &off);
}

void VulkanCommandList::bind_index_buffer(BufferHandle handle, usize offset) {
    Buffer* b = device_.buffer(handle);
    NOVA_VERIFY(b, "invalid index buffer handle");
    vkCmdBindIndexBuffer(cmd, b->buffer, offset, VK_INDEX_TYPE_UINT32);
}

void VulkanCommandList::push_constants(const void* data, u32 size) {
    NOVA_VERIFY(bound && size <= bound->push_constant_size, "push constants exceed the pipeline's range");
    vkCmdPushConstants(cmd, bound->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, size, data);
}

void VulkanCommandList::draw(u32 vertex_count, u32 first_vertex) {
    vkCmdDraw(cmd, vertex_count, 1, first_vertex, 0);
}

void VulkanCommandList::draw_indexed(u32 index_count, u32 first_index, i32 vertex_offset) {
    vkCmdDrawIndexed(cmd, index_count, 1, first_index, vertex_offset, 0);
}

} // namespace

std::unique_ptr<Device> Device::create(const DeviceDesc& desc) {
    auto device = std::make_unique<VulkanDevice>(desc);
    if (!device->init()) {
        return nullptr;
    }
    return device;
}

} // namespace nova::rhi
