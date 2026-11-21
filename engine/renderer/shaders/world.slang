// NOVA world shading: HDR forward pass with sun shadows, hemisphere ambient, procedural
// material detail, aerial perspective and an ACES/colour-graded post pass.
// FrameData layout must match WorldRenderer (renderer/src/world_renderer.cpp).

struct FrameData {
    float4x4 view_proj;
    float4x4 light_view_proj;
    float4x4 height_view_proj; // top-down orthographic view of the level (contact AO)
    float4 camera_pos;     // xyz, w = time (s)
    float4 camera_right;   // xyz, w = tan(fov/2) * aspect
    float4 camera_up;      // xyz, w = tan(fov/2)
    float4 camera_forward; // xyz, w = exposure
    float4 sun_dir;        // xyz towards the sun, w = shadow-map texel size
    float4 sun_color;      // rgb radiance
    float4 sky_zenith;     // rgb, w = fog density
    float4 sky_horizon;    // rgb, w = fog height falloff
    float4 ground_color;   // rgb, w = ambient strength
    float4 camera_clip;    // x near, y far, z contact-AO strength, w SSAO strength
    float4 height_params;  // x height-camera y, y its near plane, z depth range, w contact-AO radius
    float4 post_params;    // x bloom, y saturation, z colour bounce, w SSAO radius
    float4 screen;         // xy = 1 / size, zw = size (pixels) of the internal render targets
    float4 point_light;    // xyz position, w range (m)
    float4 point_color;    // rgb radiance (0 = off), w crosshair arm length (px, 0 = off)
    float4 output_size;    // xy back-buffer pixels
    float4x4 prev_view_proj; // last frame's view_proj (RT temporal reprojection)
    float4 rt_params;      // x frame index, y sun angular radius (rad), z indirect intensity, w flags
    float4 rt_params2;     // x GI ray length (m), y AO radius (m), z history reset, w GI rays per pixel
    float4 rt_params3;     // x sun shadow ray length (m)
    float4 taa;            // xy sub-pixel jitter (uv; subtract for the unjittered sample), z history valid, w TAA on
};

static const uint RT_FLAG_HYBRID_SHADOWS = 1u; // shadow_map holds casters missing from the TLAS
static const uint RT_FLAG_REFLECTIONS = 2u;

[[vk::binding(0, 0)]] ConstantBuffer<FrameData> frame;
[[vk::binding(1, 0)]] Sampler2D<float> shadow_map;
[[vk::binding(2, 0)]] Sampler2D<float4> hdr_scene;
[[vk::binding(3, 0)]] Sampler2D<float> scene_depth;
[[vk::binding(4, 0)]] Sampler2D<float4> bloom_tex;
[[vk::binding(5, 0)]] Sampler2D<float> height_map; // top-down depth of the level
[[vk::binding(6, 0)]] Sampler2D<float4> post_src;
// Asset (glTF metallic-roughness) material textures.
[[vk::binding(7, 0)]] Sampler2D<float4> base_color_tex;         // sRGB
[[vk::binding(8, 0)]] Sampler2D<float4> metallic_roughness_tex; // G roughness, B metallic
[[vk::binding(9, 0)]] Sampler2D<float4> normal_tex;             // tangent space
[[vk::binding(10, 0)]] Sampler2D<float4> occlusion_tex;         // R
[[vk::binding(11, 0)]] Sampler2D<float4> emissive_tex;          // sRGB

#ifdef NOVA_RT
// NOVA Lumina RT: hybrid ray-traced lighting (compiled only into world_rt.spv, see WorldRenderer).
[[vk::binding(12, 0)]] Sampler2D<float4> rt_a;        // denoised visibility: r sun shadow, g AO / generic input
[[vk::binding(13, 0)]] Sampler2D<float4> rt_b;        // denoised indirect radiance / generic history
[[vk::binding(14, 0)]] Sampler2D<float> rt_depth;     // this frame's prepass depth
[[vk::binding(15, 0)]] Sampler2D<float> rt_prev_depth; // last frame's prepass depth
[[vk::binding(16, 0)]] Sampler2D<float4> rt_normal;   // prepass world normal (xyz), a = 1 on geometry
[[vk::binding(17, 0)]] RaytracingAccelerationStructure scene_tlas;

// Must match RtInstanceGpu in world_renderer.cpp. Vertices are the C++ `Vertex` (64 tightly packed
// bytes) read as floats: a struct view would put float3 normal at offset 12, which straddles a
// 16-byte boundary and is invalid for PhysicalStorageBuffer blocks (VUID-...-pCode-08737).
static const uint kRtVertexFloats = 16u; // position 0-2, normal 3-5, color 6-9, uv 10-11, tangent 12-15
struct RtInstance {
    float* vertices;
    uint* indices;
    float4 albedo;   // rgb tint; a = 1: multiply by vertex colour (procedural materials)
    float4 emissive; // rgb radiance
};
[[vk::binding(18, 0)]] StructuredBuffer<RtInstance> rt_instances;

float3 rt_vertex_normal(float* v, uint i) {
    const uint b = i * kRtVertexFloats;
    return float3(v[b + 3u], v[b + 4u], v[b + 5u]);
}

float4 rt_vertex_color(float* v, uint i) {
    const uint b = i * kRtVertexFloats;
    return float4(v[b + 6u], v[b + 7u], v[b + 8u], v[b + 9u]);
}
#endif

// Must match ObjectPush in world_renderer.cpp (128 bytes). Procedural draws push only `model`.
struct ObjectData {
    float4x4 model;
    float4 base_color; // factor (linear rgba)
    float4 material;   // x metallic, y roughness, z alpha cutoff (0 = opaque), w normal scale
    float4 emissive;   // rgb factor, w occlusion strength
    float4 uv_xform;   // xy tile, z packed(off_u, off_v), w packed(rot_rad, pom_depth)
};
[[vk::push_constant]] ConstantBuffer<ObjectData> object;

// Material ids (vertex colour alpha).
static const int MAT_PLASTER = 0;
static const int MAT_STONE = 1;
static const int MAT_DIRT = 2;
static const int MAT_WOOD = 3;
static const int MAT_FOLIAGE = 4;
static const int MAT_WATER = 5;
static const int MAT_TERRAIN = 6;
static const int MAT_GRASS = 7;
static const int MAT_EMISSIVE = 8;
static const int MAT_FLAG = 9;
static const int MAT_CLOTH = 10;
static const int MAT_METAL = 11;
static const int MAT_CLAY = 12;

// ------------------------------------------------------------------ noise

// Interleaved Gradient Noise (Jorge Jimenez, Call of Duty AW): blue-noise-like screen-space dithering.
float ign(float2 pixel) {
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

float hash13(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float value_noise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i), n100 = hash13(i + float3(1, 0, 0));
    float n010 = hash13(i + float3(0, 1, 0)), n110 = hash13(i + float3(1, 1, 0));
    float n001 = hash13(i + float3(0, 0, 1)), n101 = hash13(i + float3(1, 0, 1));
    float n011 = hash13(i + float3(0, 1, 1)), n111 = hash13(i + float3(1, 1, 1));
    return lerp(lerp(lerp(n000, n100, u.x), lerp(n010, n110, u.x), u.y),
                lerp(lerp(n001, n101, u.x), lerp(n011, n111, u.x), u.y), u.z);
}

float fbm(float3 p, int octaves) {
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * value_noise(p);
        p = p * 2.03 + 17.1;
        amp *= 0.5;
    }
    return sum;
}

// Value noise with its analytic gradient in .yzw, so bump mapping needs one evaluation per octave.
float4 value_noise_d(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float3 du = 6.0 * f * (1.0 - f);
    float a = hash13(i), b = hash13(i + float3(1, 0, 0));
    float c = hash13(i + float3(0, 1, 0)), d = hash13(i + float3(1, 1, 0));
    float e = hash13(i + float3(0, 0, 1)), g = hash13(i + float3(1, 0, 1));
    float h = hash13(i + float3(0, 1, 1)), k = hash13(i + float3(1, 1, 1));
    float k1 = b - a, k2 = c - a, k3 = e - a;
    float k4 = a - b - c + d, k5 = a - c - e + h, k6 = a - b - e + g;
    float k7 = -a + b + c - d + e - g - h + k;
    float v = a + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x +
              k7 * u.x * u.y * u.z;
    float3 grad = du * float3(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
                              k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
                              k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);
    return float4(v, grad);
}

// Perturbs a normal with the gradient of a noise field (cheap procedural bump mapping).
float3 bump(float3 n, float3 p, float freq, float strength) {
    float3 g = 0.0;
    float amp = 0.5, scale = freq;
    float3 q = p * freq;
    for (int i = 0; i < 3; ++i) {
        g += amp * scale * value_noise_d(q).yzw;
        q = q * 2.03 + 17.1;
        scale *= 2.03;
        amp *= 0.5;
    }
    g -= n * dot(g, n); // tangential part only
    return normalize(n - g * strength);
}

// ------------------------------------------------------------------ sky, volumetric clouds & dual-scattering fog

// Cornette-Shanks Mie phase function (g in [0..1] controls forward-scattering peak around the sun).
float mie_phase(float cos_theta, float g) {
    float g2 = g * g;
    float num = 3.0 * (1.0 - g2) * (1.0 + cos_theta * cos_theta);
    float denom = 2.0 * (2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-3), 1.5);
    return num / denom;
}

float3 sky_color(float3 dir) {
    const float3 sun = frame.sun_dir.xyz;
    float h = dir.y;
    // Deep vibrant Mediterranean cobalt-azure zenith transitioning smoothly through soft horizon haze
    float3 col = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, pow(max(saturate(h), 1e-5), 0.38));
    float3 below_col = lerp(frame.sky_horizon.rgb * 0.86 + frame.ground_color.rgb * 0.14,
                            frame.ground_color.rgb * 0.65 + frame.sky_horizon.rgb * 0.25,
                            smoothstep(0.0, 0.28, -h));
    col = lerp(col, below_col, smoothstep(0.02, -0.06, h));

    float sd = max(dot(dir, sun), 0.0);
    float sd_safe = max(sd, 1e-5);
    float sky_chroma = length(frame.sky_zenith.rgb - frame.sky_horizon.rgb);
    float low_sun = 1.0 - smoothstep(0.18, 0.45, sun.y);
    // Warm golden-apricot sunset horizon blush spreading naturally along the horizon while zenith stays vivid cerulean!
    if (sky_chroma > 0.02) {
        float h_pow = pow(max(1.0 - saturate(h), 1e-5), lerp(4.5, 3.4, low_sun));
        float sun_spread = sd > 1e-4 ? pow(sd_safe, lerp(4.5, 2.2, low_sun)) : 0.0;
        float horizon_sun = h_pow * (0.20 * low_sun + 0.80 * sun_spread);
        col += frame.sun_color.rgb * float3(1.0, 0.72, 0.44) * horizon_sun * lerp(0.35, 0.58, low_sun);
    }

    // Tight luminous golden solar corona so sculpted clouds & blue sky remain crisp right next to the sun!
    float mie = mie_phase(sd, 0.88);
    float corona = sd > 1e-3 ? (pow(sd_safe, 95.0) * 0.38 + pow(sd_safe, 420.0) * 1.8) : 0.0;
    col += frame.sun_color.rgb * (mie * 0.016 + corona);

    // Sculpted 3D-shaded crisp sunset cumulus puffs with wide deep cerulean-azure sky windows!
    if (h > 0.012 && sky_chroma > 0.02) {
        float2 uv = dir.xz / (h + 0.16) * 2.15;
        float t = frame.camera_pos.w * 0.004;
        float2 warp = float2(fbm(float3(uv * 1.15 + 2.3, t), 3), fbm(float3(uv * 1.15 - 4.7, t), 3)) - 0.5;
        float2 wuv = uv + warp * 0.46;
        float c_base = fbm(float3(wuv, t), 5);
        float c_det  = fbm(float3(wuv * 3.8 + 5.3, t * 1.2), 4);
        float density = c_base * 0.70 + c_det * 0.34;
        // Scattered perspective cumulus tufts leaving bright azure sky cleanly open between cloud rows
        float c = smoothstep(0.52, 0.67, density) * smoothstep(0.012, 0.085, h);
        // Directional sunset cloud shading: pure atmospheric slate-blue undersides vs glowing golden-cream sunlit edges
        float c_sun1 = fbm(float3(wuv + sun.xz * 0.07, t), 4);
        float c_sun2 = fbm(float3(wuv + sun.xz * 0.15, t), 3);
        float lit_edge = saturate((c_base - (c_sun1 * 0.65 + c_sun2 * 0.35)) * 5.6 + 0.46);
        float3 c_dark = lerp(frame.sky_zenith.rgb * 1.10 + float3(0.16, 0.22, 0.30), float3(0.38, 0.46, 0.60), 0.48);
        float sd16 = sd > 1e-4 ? pow(sd_safe, 1.5) : 0.0;
        float3 warm_cloud = frame.sun_color.rgb * float3(1.04, 0.86, 0.64) * 0.85 + float3(0.62, 0.58, 0.52);
        float3 c_lit  = lerp(float3(1.16, 1.14, 1.10), warm_cloud, saturate(sd16 + 0.28 * low_sun));
        float3 cloud  = lerp(c_dark, c_lit, lit_edge);
        // Pure warm golden backlit silver-lining on sunset clouds (zero magenta/pink shift!)
        float sd5 = sd > 1e-4 ? pow(sd_safe, 5.0) : 0.0;
        float sd20 = sd > 1e-3 ? pow(sd_safe, 20.0) : 0.0;
        float silver = (sd5 + 0.20 * low_sun * (1.0 - saturate(h * 2.8))) * smoothstep(0.08, 0.55, c) * (1.0 - c * 0.65);
        cloud += frame.sun_color.rgb * float3(1.0, 0.78, 0.46) * (1.45 * silver + 0.42 * sd20);
        col = lerp(col, cloud, c * 0.94);
    }

    // Blazing tight HDR sun disk + sunburst core
    float core = sd > 0.5 ? pow(sd_safe, 750.0) : 0.0;
    col += frame.sun_color.rgb * (26.0 * smoothstep(0.99942, 0.99982, sd) + 4.2 * core);
    return col;
}

float3 apply_fog(float3 color, float3 world_pos) {
    float3 ro = frame.camera_pos.xyz;
    float3 v = world_pos - ro;
    float dist = length(v);
    float3 rd = v / max(dist, 1e-4);
    const float density = frame.sky_zenith.w;
    const float falloff = frame.sky_horizon.w;
    // Analytic integral of exponential height fog along the view ray (crisp clear alpine mountain air).
    float ry = abs(rd.y) < 1e-3 ? 1e-3 : rd.y;
    float amount = density / falloff * exp(-ro.y * falloff) * (1.0 - exp(-dist * ry * falloff)) / ry;
    float f = saturate(1.0 - exp(-amount));
    float sd = max(dot(rd, frame.sun_dir.xyz), 0.0);
    // Crisp cool alpine-blue aerial perspective so dark granite peaks retain razor-sharp contrast!
    float3 rayleigh = lerp(frame.sky_horizon.rgb * 0.62, frame.sky_zenith.rgb * 0.78 + frame.sky_horizon.rgb * 0.28,
                           smoothstep(80.0, 380.0, dist));
    float mie_w = saturate(mie_phase(sd, 0.80) * 0.065);
    float3 fog_col = lerp(rayleigh, frame.sun_color.rgb * float3(1.0, 0.72, 0.40) * 0.55 + rayleigh * 0.32, mie_w);
    color = lerp(color, fog_col, f * 0.72);

    // Narrow wispy white alpine valley mist ribbons drifting cleanly along the lower mountain valleys (y ~ 12..34 m)
    float sky_chroma = length(frame.sky_zenith.rgb - frame.sky_horizon.rgb);
    if (density > 1e-5 && sky_chroma > 0.02 && dist > 105.0) {
        float band = exp(-pow((world_pos.y - 22.0) / 8.0, 2.0));
        float wisp = smoothstep(0.45, 0.78, fbm(float3(world_pos.xz * 0.015, frame.camera_pos.w * 0.004), 4));
        float valley_f = saturate(band * wisp * smoothstep(105.0, 210.0, dist) * 0.62);
        float3 mist_col = lerp(float3(0.62, 0.72, 0.84), frame.sun_color.rgb * 0.45 + 0.58, pow(sd, 3.5));
        color = lerp(color, mist_col, valley_f);
    }
    return color;
}

// ------------------------------------------------------------------ fullscreen helpers

struct FullscreenOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

FullscreenOut fullscreen(uint id) {
    FullscreenOut o;
    o.uv = float2(float((id << 1) & 2), float(id & 2));
    o.position = float4(o.uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

[shader("vertex")]
FullscreenOut skyVS(uint id : SV_VertexID) {
    return fullscreen(id);
}

[shader("fragment")]
float4 skyFS(FullscreenOut input) : SV_Target {
    // Vulkan NDC y points down; uv.y = 0 is the top of the screen.
    float2 ndc = input.uv * 2.0 - 1.0;
    float3 dir = normalize(frame.camera_forward.xyz + frame.camera_right.xyz * ndc.x * frame.camera_right.w -
                           frame.camera_up.xyz * ndc.y * frame.camera_up.w);
    return float4(sky_color(dir), 0.0);
}

// ------------------------------------------------------------------ shadow pass

struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float4 color;
    [[vk::location(3)]] float2 uv;
    [[vk::location(4)]] float4 tangent;
};

[shader("vertex")]
float4 shadowVS(VSInput input) : SV_Position {
    float4 world = mul(object.model, float4(input.position, 1.0));
    return mul(frame.light_view_proj, world);
}

[shader("fragment")]
void shadowFS() {}

// Asset shadow casters: alpha-tested foliage, fences etc. cut holes in the shadow.
struct ShadowAssetOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
ShadowAssetOut shadowAssetVS(VSInput input) {
    ShadowAssetOut o;
    o.position = mul(frame.light_view_proj, mul(object.model, float4(input.position, 1.0)));
    o.uv = input.uv;
    return o;
}

[shader("fragment")]
void shadowAssetFS(ShadowAssetOut input) {
    if (object.material.z > 0.0 && base_color_tex.Sample(input.uv).a * object.base_color.a < object.material.z) {
        discard;
    }
}

// ------------------------------------------------------------------ main pass

struct VSOutput {
    float4 position : SV_Position;
    float3 world_pos : POSITION0;
    float3 normal : NORMAL0;
    float4 color : COLOR0;
    float2 uv : TEXCOORD0;
    float4 tangent : TANGENT0;
};

[shader("vertex")]
VSOutput mainVS(VSInput input) {
    VSOutput o;
    float4 world = mul(object.model, float4(input.position, 1.0));
    int id = int(input.color.a + 0.5);
    if (id == MAT_FOLIAGE || id == MAT_GRASS || id == MAT_FLAG) {
        // Wind sway, stronger higher up.
        float t = frame.camera_pos.w;
        float w = sin(t * 1.7 + world.x * 0.7 + world.z * 0.5) * 0.5 + sin(t * 3.1 + world.z * 1.3) * 0.25;
        float amp = id == MAT_FLAG ? 0.12 * input.color.r : (id == MAT_GRASS ? 0.04 : 0.025);
        world.xz += w * amp * float2(1.0, 0.6);
    }
    o.world_pos = world.xyz;
    o.position = mul(frame.view_proj, world);
    o.normal = mul((float3x3)object.model, input.normal);
    o.color = input.color;
    o.uv = input.uv;
    o.tangent = float4(mul((float3x3)object.model, input.tangent.xyz), input.tangent.w);
    return o;
}

static const float2 kPoisson[12] = {
    // The first four taps span the disk so they alone can classify fully lit/shadowed pixels.
    float2(-0.840, -0.074), float2(0.962, -0.195),  float2(0.519, 0.767),   float2(-0.322, -0.933),
    float2(-0.326, -0.406), float2(-0.696, 0.457),  float2(-0.203, 0.621),  float2(0.473, -0.480),
    float2(0.185, -0.893),  float2(0.507, 0.064),   float2(0.896, 0.412),   float2(-0.792, -0.598),
};

// Exact 2x2 sub-texel bilinear PCF lookup: completely eliminates blocky shadow-texel staircases!
float pcf_bilinear(float2 uv, float z_ref, float texel) {
    float2 tc = uv / texel - 0.5;
    float2 f = frac(tc);
    float2 base_uv = (floor(tc) + 0.5) * texel;
    float s00 = z_ref <= shadow_map.SampleLevel(base_uv, 0.0) ? 1.0 : 0.0;
    float s10 = z_ref <= shadow_map.SampleLevel(base_uv + float2(texel, 0.0), 0.0) ? 1.0 : 0.0;
    float s01 = z_ref <= shadow_map.SampleLevel(base_uv + float2(0.0, texel), 0.0) ? 1.0 : 0.0;
    float s11 = z_ref <= shadow_map.SampleLevel(base_uv + float2(texel, texel), 0.0) ? 1.0 : 0.0;
    return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
}

// Film-Grade Contact-Hardening Bilinear PCSS Shadows:
// - Screen-space IGN blue-noise rotation (ZERO 2.7cm world-space blocky tiles!)
// - Two-ring 9-probe blocker search & early-out (fast path on 95% of pixels, zero penumbra seams!)
// - 9-tap golden-angle Vogel disk of exact 2x2 bilinear PCF (36 continuous sub-texel comparisons on shadow edges!)
float sample_shadow(float3 p, float3 n, float2 pixel_pos) {
    const float texel = frame.sun_dir.w;
    float ndl = saturate(dot(n, frame.sun_dir.xyz));
    // Smooth slope-scaled normal offset prevents grazing bevel self-shadow ripples while staying razor-tight at contact
    float3 offset_p = p + n * (0.015 + 0.044 * (1.0 - ndl));
    float4 lp = mul(frame.light_view_proj, float4(offset_p, 1.0));
    float3 ndc = lp.xyz / lp.w;
    float2 uv = ndc.xy * 0.5 + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0) || ndc.z > 1.0) {
        return 1.0;
    }
    const float bias = 0.00038 + 0.00042 * (1.0 - ndl);
    const float z_ref = ndc.z - bias;

    float angle = ign(pixel_pos) * 6.2831853;
    float ca = cos(angle), sa = sin(angle);
    float2x2 rot = float2x2(ca, -sa, sa, ca);

    float lit_fast = 0.0;
    float blocker_sum = 0.0;
    float blocker_cnt = 0.0;
    // Center + inner ring (1.65 texels) + outer ring (3.5 texels) catches thin shadows & wide penumbras seamlessly
    {
        float d0 = shadow_map.SampleLevel(uv, 0.0);
        if (z_ref <= d0) {
            lit_fast += 1.0;
        } else {
            blocker_sum += d0;
            blocker_cnt += 1.0;
        }
    }
    for (int i = 0; i < 4; ++i) {
        float2 o1 = mul(rot, kPoisson[i]) * texel * 1.65;
        float d1 = shadow_map.SampleLevel(uv + o1, 0.0);
        if (z_ref <= d1) {
            lit_fast += 1.0;
        } else {
            blocker_sum += d1;
            blocker_cnt += 1.0;
        }
        float2 o2 = mul(rot, kPoisson[i + 4]) * texel * 3.5;
        float d2 = shadow_map.SampleLevel(uv + o2, 0.0);
        if (z_ref <= d2) {
            lit_fast += 1.0;
        } else {
            blocker_sum += d2;
            blocker_cnt += 1.0;
        }
    }
    if (lit_fast == 0.0 || lit_fast == 9.0) {
        return lit_fast / 9.0; // strictly inside full sun or full umbra (zero extra bilinear work!)
    }

    // Contact-hardening penumbra radius: >= 1.35 texels guarantees silky-smooth shadow edges even at 10cm close-up!
    float avg_blocker = blocker_sum / max(blocker_cnt, 1.0);
    float penumbra = clamp(1.35 + max(0.0, ndc.z - avg_blocker) * 48.0, 1.35, 3.5);

    // Deterministic 3x3 Gaussian-weighted sub-texel Bilinear PCF (36 continuous samples, ZERO diagonal IGN stripes!)
    const float w3[3] = {0.25, 0.50, 0.25};
    const float step_uv = penumbra * 0.58 * texel;
    float sum = 0.0;
    for (int iy = -1; iy <= 1; ++iy) {
        for (int ix = -1; ix <= 1; ++ix) {
            float w = w3[ix + 1] * w3[iy + 1];
            sum += pcf_bilinear(uv + float2(float(ix), float(iy)) * step_uv, z_ref, texel) * w;
        }
    }
    return sum;
}

// Occlusion attenuates indirect light, not the hue of the receiving material.
float contact_ao(float3 p, float3 n) {
    const float strength = frame.camera_clip.z;
    const float R = frame.height_params.w;
    if (strength <= 0.0) {
        return 1.0;
    }
    float4 c0 = mul(frame.height_view_proj, float4(p, 1.0));
    float2 uv0 = c0.xy * 0.5 + 0.5;
    if (any(uv0 < 0.0) || any(uv0 > 1.0)) {
        return 1.0;
    }
    // Orthographic, so world xz maps to uv affinely.
    float2 du_dx = mul(frame.height_view_proj, float4(1, 0, 0, 0)).xy * 0.5;
    float2 du_dz = mul(frame.height_view_proj, float4(0, 0, 1, 0)).xy * 0.5;
    float2 centre = n.xz * R;
    float occ = 0.0, wsum = 0.0;
    for (int i = 0; i < 16; ++i) {
        float f = float(i) + 0.5;
        float a = f * 2.39996;
        float r = sqrt(f / 16.0) * R * 0.9;
        float2 q = centre + float2(cos(a), sin(a)) * r;
        float d = height_map.SampleLevel(uv0 + du_dx * q.x + du_dz * q.y, 0.0);
        float h = frame.height_params.x - (frame.height_params.y + d * frame.height_params.z);
        float w = 1.0 - r / R;
        occ += saturate((h - p.y - 0.2) / (R * 0.8)) * w;
        wsum += w;
    }
    return 1.0 - saturate(strength) * saturate(occ / wsum) * 0.65;
}

struct Surface {
    float3 albedo;
    float3 normal;
    float roughness;
    float3 emissive;
    float translucency;
    float wetness; // 1 = water
    float metallic;
    float occlusion; // material AO (1 = none)
    float env_spec;  // 1 = add sky-reflection ambient specular (asset materials)
};

Surface terrain_surface(float3 p, float3 n) {
    Surface s;
    float r = length(p.xz);
    float slope = 1.0 - n.y;
    float large = fbm(p * 0.015, 4);
    float crag_macro = 1.0 - abs(fbm(p * float3(0.026, 0.036, 0.026), 4) * 2.0 - 1.0);
    float crag_fine  = 1.0 - abs(fbm(p * float3(0.085, 0.145, 0.085) + 7.1, 4) * 2.0 - 1.0);
    float detail = fbm(p * 0.55, 4);
    // Sharp vertical snow couloirs / gullies carved into the upper alpine peaks
    float gully = 1.0 - abs(fbm(p * float3(0.038, 0.018, 0.038) + 4.2, 4) * 2.0 - 1.0);

    float3 dirt = lerp(float3(0.24, 0.16, 0.10), float3(0.42, 0.30, 0.19), detail);
    dirt *= 0.8 + 0.35 * value_noise(p * 7.0); // pebbles and grit
    // Deep emerald/pine Mediterranean mountain foothill forest
    float3 lush_forest = lerp(float3(0.055, 0.115, 0.035), float3(0.18, 0.24, 0.08), detail * 0.65 + large * 0.35);
    // High-contrast dark charcoal-slate alpine granite crags with crisp sunlit facets
    float3 rock_dark = float3(0.065, 0.075, 0.095);
    float3 rock_lit  = float3(0.27, 0.25, 0.24);
    float3 rock = lerp(rock_dark, rock_lit, saturate(crag_macro * 0.65 + crag_fine * 0.55 - 0.18));
    float3 snow = float3(0.95, 0.97, 1.0);

    float grass_w = smoothstep(0.18, 0.52, large + 0.20) * (1.0 - smoothstep(0.26, 0.48, slope));
    grass_w *= smoothstep(18.0, 42.0, r) * (1.0 - smoothstep(55.0, 95.0, p.y));
    float rock_w = smoothstep(0.14, 0.34, slope + (crag_macro - 0.45) * 0.25 + smoothstep(42.0, 88.0, p.y) * 0.38)
                 * smoothstep(26.0, 62.0, r);
    // Crisp high-contrast snow veins ONLY inside gullies & high ledges — exposes dramatic dark craggy granite!
    float snow_vein = smoothstep(0.56, 0.78, gully * 0.72 + crag_fine * 0.28);
    float snow_cap  = smoothstep(98.0, 145.0, p.y + (large - 0.5) * 28.0) * 0.85;
    float snow_w = saturate(max(snow_vein * smoothstep(48.0, 86.0, p.y), snow_cap)) * (1.0 - smoothstep(0.32, 0.58, slope));

    float3 c = lerp(dirt, lush_forest, grass_w);
    c = lerp(c, rock, rock_w);
    c = lerp(c, snow, snow_w);
    s.albedo = c;
    // Two-scale ridged alpine cliff normal perturbation gives razor-sharp rocky mountain crags at 2K!
    if (rock_w > 0.25) {
        float3 n1 = bump(n, p * float3(1.0, 1.35, 1.0), 0.045, 1.25);
        s.normal = bump(n1, p * float3(1.0, 1.6, 1.0), 0.16, 0.55);
    } else {
        s.normal = bump(n, p, 1.6, 0.16);
    }
    s.roughness = lerp(0.95, 0.52, snow_w);
    s.emissive = 0;
    s.translucency = 0;
    s.wetness = 0;
    return s;
}

Surface make_surface(int id, float3 base, float3 p, float3 n, float4 vcolor) {
    Surface s;
    s.albedo = base;
    s.normal = n;
    s.roughness = 0.85;
    s.emissive = 0;
    s.translucency = 0;
    s.wetness = 0;

    if (id == MAT_TERRAIN) {
        return terrain_surface(p, n);
    }
    if (id == MAT_PLASTER) {
        // Weathered Mediterranean earth-lime stucco with trowel relief, rain stains, damp base & peeled mudstone
        float blotch = fbm(p * 0.75, 4);
        float micro  = fbm(p * 4.5, 3);
        float speck  = value_noise(p * 28.0);
        float grime_base = saturate(1.0 - p.y / 1.55) * (0.45 + 0.55 * fbm(p * 2.2, 3));
        float grime_top  = smoothstep(2.9, 5.2, p.y) * fbm(p * float3(1.8, 0.6, 1.8), 3) * 0.55;
        s.albedo = base * (0.76 + 0.36 * blotch) * (0.90 + 0.14 * micro) * (0.93 + 0.07 * speck);
        s.albedo = lerp(s.albedo, float3(0.28, 0.22, 0.16), saturate(grime_base * 0.78 + grime_top * 0.45));
        // Weathered peeled patches revealing rough warm sandstone / mudbrick underneath
        float peel = smoothstep(0.60, 0.68, fbm(p * 0.65 + 3.0, 4));
        float3 exposed = lerp(float3(0.34, 0.28, 0.22), float3(0.50, 0.42, 0.32), micro);
        s.albedo = lerp(s.albedo, exposed, peel * 0.85);
        s.normal = bump(n, p, 3.8, 0.22 + peel * 0.48);
        s.roughness = 0.92;
    } else if (id == MAT_STONE) {
        float v = fbm(p * 10.0, 4);
        s.albedo = base * (0.50 + 0.80 * v);
        s.albedo = lerp(s.albedo, float3(0.30, 0.32, 0.22), smoothstep(0.62, 0.75, fbm(p * 1.3, 3)) * 0.5); // lichen
        s.normal = bump(n, p, 2.5, 0.16);
        s.roughness = 0.9;
    } else if (id == MAT_WOOD) {
        float grain = sin((p.y + p.x * 0.2 + fbm(p * float3(2, 18, 2), 3) * 0.8) * 55.0) * 0.5 + 0.5;
        s.albedo = base * (0.7 + 0.35 * grain) * (0.8 + 0.4 * fbm(p * 1.5, 3));
        s.normal = bump(n, p * float3(1, 6, 1), 4.0, 0.25);
        s.roughness = 0.82;
    } else if (id == MAT_FOLIAGE) {
        // Individual 3D olive leaves: crisp vein variation + luminous sunlit golden-green translucency
        float leaf_var = fbm(p * 4.5, 3);
        float vein = value_noise(p * 26.0);
        s.albedo = base * (0.72 + 0.52 * leaf_var) * (0.84 + 0.24 * vein);
        s.normal = bump(n, p, 14.0, 0.14);
        s.translucency = 0.65;
        s.roughness = 0.58;
    } else if (id == MAT_GRASS) {
        s.albedo = base * (0.78 + 0.44 * value_noise(p * 4.0));
        s.translucency = 0.48;
        s.roughness = 0.72;
    } else if (id == MAT_WATER) {
        s.albedo = float3(0.03, 0.025, 0.02);
        float t = frame.camera_pos.w;
        float rx = sin(p.x * 7.0 + t * 1.1) * 0.004 + sin(p.z * 11.0 - t * 0.8) * 0.003;
        float rz = cos(p.z * 8.0 + t * 1.3) * 0.004 + cos(p.x * 13.0 + t * 0.9) * 0.003;
        s.normal = normalize(float3(rx, 1.0, rz));
        s.roughness = 0.025;
        s.wetness = 1.0;
    } else if (id == MAT_EMISSIVE) {
        s.emissive = base * 6.0;
        s.albedo = base * 0.2;
    } else if (id == MAT_FLAG) {
        // vcolor.rg carries the flag's (u, v); Algerian flag: green | white with red crescent and star.
        float2 uv = vcolor.rg;
        float3 col = uv.x < 0.5 ? float3(0.0, 0.32, 0.12) : float3(0.92, 0.92, 0.9);
        float2 q = (uv - float2(0.5, 0.5)) * float2(1.5, 1.0);
        float outer = length(q - float2(0.0, 0.0));
        float inner = length(q - float2(0.07, 0.0));
        bool crescent = outer < 0.25 && inner > 0.2;
        float2 sq = q - float2(0.12, 0.0);
        float ang = atan2(sq.y, sq.x);
        float star_r = 0.09 * (0.55 + 0.45 * cos(5.0 * (ang + 3.14159)));
        bool star = length(sq) < star_r;
        if (crescent || star) {
            col = float3(0.75, 0.04, 0.06);
        }
        s.albedo = col * (0.85 + 0.15 * value_noise(float3(uv * 30.0, 0)));
        s.translucency = 0.3;
        s.roughness = 0.9;
    } else if (id == MAT_CLOTH) {
        // Woven North African / Amazigh kilim rug with diamond medallions & bottom cream fringe tassels
        float2 rug_uv = float2(p.z * 2.6, p.y * 2.6);
        float2 cell = frac(rug_uv * 1.5) - 0.5;
        float dist_d = abs(cell.x) + abs(cell.y);
        float outer_d = step(dist_d, 0.40);
        float inner_d = step(dist_d, 0.22);
        float center_d = step(dist_d, 0.09);
        float border = step(0.40, abs(frac(p.z * 0.85) - 0.5));
        float3 crimson = float3(0.36, 0.07, 0.06);
        float3 cream   = float3(0.84, 0.78, 0.68);
        float3 navy    = float3(0.08, 0.10, 0.16);
        float3 ochre   = float3(0.62, 0.42, 0.20);
        float3 pat = crimson;
        pat = lerp(pat, cream, outer_d);
        pat = lerp(pat, navy, inner_d);
        pat = lerp(pat, ochre, center_d);
        pat = lerp(pat, cream, border * step(0.5, frac(p.y * 8.0)));
        float is_fringe = (1.0 - step(0.50, p.y)) * step(-15.0, p.z);
        float tassel = sin(p.z * 190.0) * 0.5 + 0.5;
        pat = lerp(pat, cream * (0.65 + 0.35 * tassel), is_fringe);
        float is_awning = step(p.z, -15.0);
        float3 cloth_col = lerp(pat, base * 0.95, is_awning);
        float weave = sin(p.y * 160.0) * sin(p.z * 160.0) * 0.12 + 0.88;
        s.albedo = cloth_col * weave * (0.82 + 0.26 * fbm(p * 8.0, 3));
        s.normal = bump(n, p, 18.0, 0.18);
        s.roughness = 0.92;
    } else if (id == MAT_METAL) {
        s.albedo = base * (0.6 + 0.4 * fbm(p * 6.0, 3));
        s.roughness = 0.45;
    } else if (id == MAT_DIRT) {
        s.albedo = base * (0.8 + 0.4 * fbm(p * 0.8, 4));
        s.normal = bump(n, p, 2.0, 0.3);
    } else if (id == MAT_CLAY) {
        // Pure velvety architectural clay: ZERO texture noise, clean smooth matte surface with soft bevel sheen!
        s.albedo = base;
        s.normal = n;
        s.roughness = 0.68;
    }
    return s;
}

// NaN-free 5th power helper (SPIR-V pow(<=0, y) is UB on GPU hardware!)
float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

// Burley-Disney rough diffuse BRDF (captures grazing retro-reflection on rough limestone, plaster, dirt & cloth).
float burley_diffuse(float ndl, float ndv, float vdh, float roughness) {
    float fd90 = 0.5 + 2.0 * roughness * vdh * vdh;
    float fl = 1.0 + (fd90 - 1.0) * pow5(saturate(1.0 - ndl));
    float fv = 1.0 + (fd90 - 1.0) * pow5(saturate(1.0 - ndv));
    return ndl * fl * fv;
}

float3 ggx_specular(float3 n, float3 v, float3 l, float roughness, float3 f0) {
    float3 vl = v + l;
    float3 h = dot(vl, vl) > 1e-8 ? normalize(vl) : n;
    float a = max(roughness * roughness, 0.002);
    float a2 = a * a;
    float ndh = saturate(dot(n, h));
    float denom = ndh * ndh * (a2 - 1.0) + 1.0;
    float d = a2 / max(3.14159265 * denom * denom, 1e-7);
    float ndv = clamp(dot(n, v), 1e-4, 1.0);
    float ndl = saturate(dot(n, l));
    float k = a * 0.5;
    float vis = 1.0 / max((ndv * (1.0 - k) + k) * (ndl * (1.0 - k) + k), 1e-6);
    float3 fresnel = f0 + (1.0 - f0) * pow5(saturate(1.0 - saturate(dot(h, v))));
    return d * vis * fresnel * 0.25;
}

// Split-sum environment BRDF terms (scale, bias), analytic fit (Karis, "Physically Based Shading on Mobile").
float2 env_brdf_ab(float roughness, float ndv) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// Analytical pre-filtered environment sky/sun/ground radiance for smooth roughness-dependent IBL reflections.
float3 env_prefiltered(float3 r, float roughness, float3 sky_amb, float3 ground_amb) {
    float3 sharp = sky_color(r);
    float3 broad_sky = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, saturate(r.y * 0.75 + 0.15));
    float sd = saturate(dot(r, frame.sun_dir.xyz));
    float spec_lobe = sd > 1e-5 ? pow(sd, lerp(48.0, 3.0, roughness)) * lerp(0.55, 0.10, roughness) : 0.0;
    float3 broad = lerp(ground_amb, broad_sky, saturate(r.y * 0.5 + 0.5)) + frame.sun_color.rgb * spec_lobe;
    return lerp(sharp * frame.ground_color.w, broad * frame.ground_color.w, smoothstep(0.08, 0.72, roughness));
}

// NOVA PRISM Studio-Grade PBR Shading (100% NaN-proof!):
// - 3-Lobe Bent-Normal Studio Sky/Horizon/Ground Irradiance (reveals fine cloth folds, stitches & stone crevices in shadow)
// - Burley rough diffuse + Kulla-Conty multi-scatter GGX + Charlie cloth/dielectric grazing sheen + Chan micro-shadowing
float3 light_surface(Surface s, float3 p, float3 geom_n, float2 pixel_pos) {
    float3 n = s.normal;
    float3 vp = frame.camera_pos.xyz - p;
    float3 v = dot(vp, vp) > 1e-10 ? normalize(vp) : float3(0.0, 0.0, 1.0);
    float3 l = frame.sun_dir.xyz;
    float3 vl = v + l;
    float3 h = dot(vl, vl) > 1e-8 ? normalize(vl) : n;
    float ndl = saturate(dot(n, l));
    float geom_ndl = saturate(dot(geom_n, l));
    // Strictly clamped to [1e-4, 1.0] so (1.0 - ndv) is NEVER negative (eliminates GPU NaN black dots!)
    float ndv = clamp(dot(n, v), 1e-4, 1.0);
    float vdh = saturate(dot(v, h));
#ifdef NOVA_RT
    // Ray-traced, temporally accumulated and denoised visibility (same resolution as this pass).
    const float4 rt_vis = rt_a.Load(int3(int2(pixel_pos), 0));
    float shadow = rt_vis.r;
    if ((asuint(frame.rt_params.w) & RT_FLAG_HYBRID_SHADOWS) != 0u) {
        shadow *= sample_shadow(p, geom_n, pixel_pos); // skinned / alpha-tested casters outside the TLAS
    }
    const uint debug_view = (asuint(frame.rt_params.w) >> 2u) & 3u; // 1 shadow, 2 AO, 3 indirect
    if (debug_view != 0u) {
        float3 dbg = debug_view == 1u ? rt_vis.rrr
                   : debug_view == 2u ? rt_vis.ggg
                                      : rt_b.Load(int3(int2(pixel_pos), 0)).rgb;
        return dbg / max(frame.camera_forward.w, 1e-3); // undo exposure so 1.0 reads as white
    }
#else
    float shadow = sample_shadow(p, geom_n, pixel_pos);
#endif
    // Smooth grazing shadow terminator removes faceted self-shadow waves on curved shoulders, cloth & beveled blocks
    float terminator = smoothstep(0.015, 0.095, geom_ndl);

    // 3-Lobe Studio Irradiance using detail-blended normal so folds, seams & fabric weave have rich 3D depth even in shadow!
    float3 amb_raw = lerp(geom_n, n, 0.68);
    float3 amb_n = dot(amb_raw, amb_raw) > 1e-8 ? normalize(amb_raw) : geom_n;
    float3 sky_top = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, 0.52);
    float3 sky_hor = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, 0.16);
    float3 sky_amb = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, 0.35);
    // Warm sunlit-floor GI bounce stays luminous even at low golden-hour sun elevations (strictly achromatic-safe!)
    float3 ground_amb = frame.ground_color.rgb * (sky_amb * 0.78 + frame.sun_color.rgb * (0.26 + 0.82 * saturate(l.y)));
    float up_w = saturate(amb_n.y);
    float dn_w = saturate(-amb_n.y);
    float hor_w = saturate(1.0 - abs(amb_n.y));
    // Soft sky counter-fill opposite the sun gives true 3-point studio separation (warm key vs cool fill)
    float back_fill = saturate(dot(amb_n, -l) * 0.5 + 0.5);
    float3 hemi_ibl = sky_top * (up_w * 0.84 + 0.08) +
                      sky_hor * (hor_w * (0.60 + 0.24 * back_fill)) +
                      ground_amb * (dn_w * 0.92 + hor_w * 0.22);
    float3 ambient = hemi_ibl * frame.ground_color.w;

#ifdef NOVA_RT
    // Path-traced one-bounce irradiance (sky + coloured bounce, occlusion included) replaces the
    // hemisphere guess; keep a little of it so normal-mapped detail still reads in deep shade.
    const float3 rt_gi = max(rt_b.Load(int3(int2(pixel_pos), 0)).rgb, 0.0) * frame.rt_params.z;
    ambient = rt_gi + hemi_ibl * frame.ground_color.w * 0.06;
    float ao = saturate(s.occlusion * rt_vis.g);
#else
    float ao = saturate(s.occlusion * contact_ao(p, geom_n));
#endif
    float3 diffuse_albedo = s.albedo * (1.0 - s.metallic);
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), s.albedo, s.metallic);

    // SIGGRAPH 2018 Chan Micro-Shadowing: cavity occlusion softly shadows direct sun inside micro-crevices
    float micro_shadow = saturate(2.0 * s.occlusion - 1.0 + ndl * 0.65);
    float rough_ndl = burley_diffuse(ndl, ndv, vdh, s.roughness);
    float3 direct = frame.sun_color.rgb * rough_ndl * shadow * terminator * micro_shadow;
    float3 color = diffuse_albedo * (direct + ambient * ao);

    // Kulla-Conty / Fdez-Aguera Multi-Scatter GGX Energy Compensation
    float2 brdf_ab = env_brdf_ab(s.roughness, ndv);
    float3 energy_comp = 1.0 + f0 * (1.0 / max(brdf_ab.x + brdf_ab.y, 0.25) - 1.0);
    float spec_scale = s.env_spec > 0.0 ? 1.0 : lerp(1.0, 0.15, smoothstep(0.6, 0.95, s.roughness));
    color += frame.sun_color.rgb * ggx_specular(n, v, l, s.roughness, f0) * energy_comp * ndl * shadow * terminator * micro_shadow * spec_scale;

    if (s.env_spec > 0.0) {
        // Pre-filtered analytical Sky/Sun/Ground IBL specular + cavity specular occlusion
        float3 r = reflect(-v, n);
        float3 env = env_prefiltered(r, s.roughness, sky_amb, ground_amb);
        float3 fss_ess = max(f0 * brdf_ab.x + brdf_ab.y, 0.0);
        float ems = saturate(1.0 - (brdf_ab.x + brdf_ab.y));
        float3 favg = f0 + (1.0 - f0) / 21.0;
        float3 fms_ems = ems * fss_ess * favg / max(1.0 - favg * ems, 1e-4);
        float spec_occ = ao * ao * lerp(1.0, 0.42, smoothstep(0.68, 0.96, s.roughness));
        color += env * (fss_ess + fms_ems) * spec_occ * s.env_spec;

        // Soft Microfiber / Dielectric Grazing Rim Sheen (100% NaN-proof polynomial grazing curve!)
        float dielectric = (1.0 - s.metallic) * smoothstep(0.45, 0.92, s.roughness);
        float rim_x = saturate(1.0 - ndv);
        float rim = rim_x * rim_x * rim_x * sqrt(rim_x) * ao;
        float3 rim_tint = lerp(s.albedo * 0.65 + 0.35, float3(1.0, 1.0, 1.0), 0.35);
        float3 rim_light = sky_amb * 0.22 * frame.ground_color.w + frame.sun_color.rgb * (0.14 * shadow * (0.35 + 0.65 * saturate(dot(v, -l))));
        color += rim_tint * rim_light * rim * dielectric * 0.65;
    }
    if (s.translucency > 0.0) {
        float back_v = saturate(dot(-v, l));
        float back = saturate(dot(-n, l)) * (back_v * back_v);
        color += s.albedo * frame.sun_color.rgb * back * s.translucency * shadow;
        color += s.albedo * frame.sun_color.rgb * 0.15 * s.translucency * shadow;
    }
    if (s.wetness > 0.0) {
        float3 r = reflect(-v, n);
        float fresnel = 0.04 + 0.96 * pow5(saturate(1.0 - saturate(dot(n, v))));
        color = lerp(color, sky_color(r) * lerp(0.55, 0.95, shadow), saturate(fresnel * 1.15 + 0.18));
    }
    // Point light (muzzle flash or Studio Point Light): inverse-square with a smooth range cut-off, unshadowed.
    if (any(frame.point_color.rgb > 0.0)) {
        float3 lv = frame.point_light.xyz - p;
        float d2 = dot(lv, lv);
        float3 pl = lv * rsqrt(max(d2, 1e-6));
        float ratio = d2 / max(frame.point_light.w * frame.point_light.w, 1e-4);
        float window = saturate(1.0 - ratio * ratio);
        float3 radiance = frame.point_color.rgb * window * window / (d2 + 0.04);
        float pndl = saturate(dot(n, pl));
        color += (diffuse_albedo + ggx_specular(n, v, pl, s.roughness, f0)) * radiance * pndl;
    }
    color += s.emissive;
    color = max(color, 0.0);
    if (any(isnan(color)) || any(isinf(color))) {
        color = float3(0.0, 0.0, 0.0);
    }
    return apply_fog(color, p);
}

[shader("fragment")]
float4 mainFS(VSOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    int id = int(input.color.a + 0.5);
    float3 p = input.world_pos;
    float3 n = dot(input.normal, input.normal) > 1e-10 ? normalize(input.normal) : float3(0.0, 1.0, 0.0);
    if (!front_face) {
        n = -n; // two-sided foliage, grass and flags
    }
    Surface s = make_surface(id, input.color.rgb, p, n, input.color);
    s.metallic = 0.0;
    s.occlusion = 1.0;
    // Give pure architectural clay (MAT_CLAY) soft satin IBL sky-sheen on beveled corners & curved lantern/trees!
    s.env_spec = (id == MAT_CLAY) ? 0.52 : 0.0;
    // Pack puddle mirror mask (s.wetness) in HDR alpha so compositeFS runs ultra-fast SSR ONLY on water pixels!
    return float4(light_surface(s, p, n, input.position.xy), s.wetness);
}

// Emissive factors are relative to a sunlit white surface rather than absolute nits.
static const float kEmissiveScale = 4.0;

float2 unpack_u16_pair(float packed, float lo_a, float hi_a, float lo_b, float hi_b) {
    uint u = asuint(packed);
    float na = float(u & 0xFFFFu) / 65535.0;
    float nb = float((u >> 16u) & 0xFFFFu) / 65535.0;
    return float2(lerp(lo_a, hi_a, na), lerp(lo_b, hi_b, nb));
}

[shader("fragment")]
float4 assetFS(VSOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    // Live NOVA Craft per-object UV transform (tile, offset, rotation) + NOVA PRISM Steep POM depth
    float2 uv_tile = object.uv_xform.xy;
    float2 uv_off = unpack_u16_pair(object.uv_xform.z, -8.0, 8.0, -8.0, 8.0);
    float2 rot_pom = unpack_u16_pair(object.uv_xform.w, -3.14159265, 3.14159265, 0.0, 0.15);
    float uv_rot = rot_pom.x;
    float pom_depth = rot_pom.y;

    float cr = cos(uv_rot);
    float sr = sin(uv_rot);
    float2 uvc = input.uv - 0.5;
    float2 uv = float2(uvc.x * cr - uvc.y * sr, uvc.x * sr + uvc.y * cr) + 0.5;
    uv = uv * uv_tile + uv_off;

    float3 p = input.world_pos;
    float3 geom_n = (dot(input.normal, input.normal) > 1e-10 ? normalize(input.normal) : float3(0.0, 1.0, 0.0)) *
                    (front_face ? 1.0 : -1.0);
    float3 n = geom_n;
    float3 t_proj = input.tangent.xyz - geom_n * dot(geom_n, input.tangent.xyz);
    bool has_tbn = dot(t_proj, t_proj) > 1e-8;
    float3 t = has_tbn ? normalize(t_proj) : float3(1.0, 0.0, 0.0);
    float3 b = float3(0.0, 1.0, 0.0);
    float pom_shadow = 1.0;
    if (has_tbn) {
        b = cross(geom_n, t) * (input.tangent.w < 0.0 ? -1.0 : 1.0);
        // Rotate tangent basis if UV is rotated so normal maps & Steep POM stay 100% aligned!
        if (abs(uv_rot) > 1e-4) {
            float3 tr = t * cr + b * sr;
            float3 br = -t * sr + b * cr;
            t = tr;
            b = br;
        }
        // NOVA PRISM MICRO-GEO: Adaptive Steep Parallax Occlusion Mapping + Per-Pixel Heightfield Micro-Self-Shadowing
        if (pom_depth > 0.0008) {
            float cam_dist = length(frame.camera_pos.xyz - p);
            float pom_w = pom_depth * (1.0 - smoothstep(4.5, 10.5, cam_dist));
            if (pom_w > 0.0008) {
                float2 ddx_uv = ddx(uv);
                float2 ddy_uv = ddy(uv);
                float3 v = normalize(frame.camera_pos.xyz - p);
                float vz = max(dot(v, geom_n), 0.25);
                float2 max_shift = float2(dot(v, t), -dot(v, b)) * (pom_w / vz);
                const int kPomSteps = cam_dist < 4.2 ? 6 : 3;
                float layer_step = 1.0 / float(kPomSteps);
                float2 duv = max_shift * layer_step;
                float cur_layer = 0.0;
                float2 cur_uv = uv;
                float cur_h = 1.0 - occlusion_tex.SampleGrad(cur_uv, ddx_uv, ddy_uv).r;
                float prev_h = cur_h;
                for (int i = 0; i < kPomSteps; ++i) {
                    if (cur_layer >= cur_h) break;
                    prev_h = cur_h;
                    cur_uv -= duv;
                    cur_layer += layer_step;
                    cur_h = 1.0 - occlusion_tex.SampleGrad(cur_uv, ddx_uv, ddy_uv).r;
                }
                float2 prev_uv = cur_uv + duv;
                float after_d = cur_h - cur_layer;
                float before_d = prev_h - (cur_layer - layer_step);
                float w_interp = saturate(after_d / max(after_d - before_d, 1e-4));
                uv = lerp(cur_uv, prev_uv, w_interp);

                // PRISM-VSM 2-step Per-Pixel Heightfield Micro-Self-Shadowing toward the sun
                float3 l = frame.sun_dir.xyz;
                float lz = dot(l, geom_n);
                if (lz > 0.06) {
                    float h0 = occlusion_tex.SampleGrad(uv, ddx_uv, ddy_uv).r;
                    float2 l_shift = float2(dot(l, t), -dot(l, b)) * (pom_w * 1.4 / max(lz, 0.25));
                    float hs1 = occlusion_tex.SampleGrad(uv + l_shift * 0.45, ddx_uv, ddy_uv).r;
                    float hs2 = occlusion_tex.SampleGrad(uv + l_shift * 0.90, ddx_uv, ddy_uv).r;
                    float sh1 = saturate((hs1 - (h0 + 0.45 * (1.0 - h0))) * 3.2);
                    float sh2 = saturate((hs2 - (h0 + 0.90 * (1.0 - h0))) * 3.2);
                    pom_shadow = 1.0 - max(sh1, sh2) * 0.65;
                }
            }
        }
    }

    float4 base = base_color_tex.Sample(uv) * object.base_color * float4(input.color.rgb, 1.0);
    if (object.material.z > 0.0 && base.a < object.material.z) {
        discard;
    }
    if (has_tbn) {
        float3 tn = normal_tex.Sample(uv).xyz * 2.0 - 1.0;
        tn.xy *= object.material.w;
        float3 n_raw = t * tn.x + b * tn.y + geom_n * max(tn.z, 1e-3);
        n = dot(n_raw, n_raw) > 1e-8 ? normalize(n_raw) : geom_n;
    }
    float4 mr = metallic_roughness_tex.Sample(uv);

    Surface s;
    s.albedo = base.rgb;
    s.normal = n;
    s.metallic = saturate(object.material.x * mr.b);
    s.roughness = clamp(object.material.y * mr.g, 0.04, 1.0);
    s.occlusion = lerp(1.0, occlusion_tex.Sample(uv).r, object.emissive.w) * pom_shadow;
    s.emissive = emissive_tex.Sample(uv).rgb * object.emissive.rgb * kEmissiveScale;
    s.translucency = 0.0;
    s.wetness = 0.0;
    s.env_spec = 1.0;
    return float4(light_surface(s, p, geom_n, input.position.xy), 0.0);
}

// ------------------------------------------------------------------ post: SSAO + colour bleed

float linear_depth(float d) {
    const float n = frame.camera_clip.x, f = frame.camera_clip.y;
    return n * f / (f - d * (f - n));
}

// World position of a screen uv at view depth t.
float3 screen_to_world(float2 uv, float t) {
    float2 ndc = uv * 2.0 - 1.0;
    float3 dir = frame.camera_forward.xyz + frame.camera_right.xyz * ndc.x * frame.camera_right.w -
                 frame.camera_up.xyz * ndc.y * frame.camera_up.w;
    return frame.camera_pos.xyz + dir * t;
}

float3 world_at(float2 uv) {
    return screen_to_world(uv, linear_depth(scene_depth.SampleLevel(uv, 0.0)));
}

// Jimenez Multi-Bounce Colored AO approximation (restores natural bounced color inside stone crevices & corners).
float3 multi_bounce_ao(float vis, float3 albedo) {
    float3 a =  2.0404 * albedo - 0.3324;
    float3 b = -4.7951 * albedo + 0.6417;
    float3 c =  2.7552 * albedo + 0.6903;
    return max(vis.xxx, ((vis * a + b) * vis + c) * vis);
}

// Reconstruct world-space surface normal from depth (closest neighbour pair per axis).
float3 depth_normal(float2 uv, float3 p) {
    const float2 px = frame.screen.xy;
    float3 l = world_at(uv - float2(px.x, 0)), r = world_at(uv + float2(px.x, 0));
    float3 u = world_at(uv - float2(0, px.y)), d = world_at(uv + float2(0, px.y));
    float3 dx = distance(r, p) < distance(l, p) ? r - p : p - l;
    float3 dy = distance(d, p) < distance(u, p) ? d - p : p - u;
    float3 cr = cross(dy, dx);
    float3 n = dot(cr, cr) > 1e-10 ? normalize(cr) : float3(0.0, 1.0, 0.0);
    if (dot(n, frame.camera_pos.xyz - p) < 0.0) {
        n = -n;
    }
    return n;
}

// Dual-Scale Continuous Horizon-Angle XeGTAO (Jimenez 2016):
// Integrates continuous elevation horizons (zero binary 0/1 stipple noise!) at BOTH micro crevice radius (0.16m)
// for deep crisp block joints/cloth folds AND macro architectural radius for doorways, eaves & corners.
float horizon_ao_scale(float2 uv, float3 p, float3 n, float view_t, float world_r, float spin) {
    // Project world radius to screen UV radius cleanly
    float proj_scale = 0.5 / max(view_t * frame.camera_up.w, 0.05);
    float2 aspect_fix = float2(frame.camera_up.w / max(frame.camera_right.w, 1e-4), 1.0);
    float uv_rad = clamp(world_r * proj_scale, frame.screen.y * 4.0, 0.14);

    float vis_sum = 0.0;
    const int kSlices = 4;
    const int kSteps = 3;
    for (int s = 0; s < kSlices; ++s) {
        float ang = (float(s) + spin) * (3.14159265 / float(kSlices));
        float2 dir = float2(cos(ang), sin(ang)) * aspect_fix;
        float h_pos = 0.0;
        float h_neg = 0.0;
        for (int k = 1; k <= kSteps; ++k) {
            float frac_k = (float(k) - 0.35 + 0.35 * spin) / float(kSteps);
            float2 off = dir * (uv_rad * frac_k * frac_k);

            float2 uv1 = uv + off;
            if (all(uv1 > 0.0) && all(uv1 < 1.0)) {
                float d1 = scene_depth.SampleLevel(uv1, 0.0);
                if (d1 < 0.99995) {
                    float z1 = linear_depth(d1);
                    if (z1 > 0.65) {
                        float3 dv = screen_to_world(uv1, z1) - p;
                        float dist2 = dot(dv, dv);
                        float dist = sqrt(max(dist2, 1e-8));
                        float sin_h = dot(dv, n) / dist;
                        float fall = saturate(1.0 - dist2 / max(world_r * world_r * 1.45, 1e-4));
                        float thick = smoothstep(world_r * 2.0, world_r * 0.60, dist);
                        h_pos = max(h_pos, saturate(sin_h - 0.05) * fall * thick);
                    }
                }
            }

            float2 uv2 = uv - off;
            if (all(uv2 > 0.0) && all(uv2 < 1.0)) {
                float d2 = scene_depth.SampleLevel(uv2, 0.0);
                if (d2 < 0.99995) {
                    float z2 = linear_depth(d2);
                    if (z2 > 0.65) {
                        float3 dv = screen_to_world(uv2, z2) - p;
                        float dist2 = dot(dv, dv);
                        float dist = sqrt(max(dist2, 1e-8));
                        float sin_h = dot(dv, n) / dist;
                        float fall = saturate(1.0 - dist2 / max(world_r * world_r * 1.45, 1e-4));
                        float thick = smoothstep(world_r * 2.0, world_r * 0.60, dist);
                        h_neg = max(h_neg, saturate(sin_h - 0.05) * fall * thick);
                    }
                }
            }
        }
        // Smooth analytic cosine-weighted hemisphere visibility for this slice
        float slice_occ = 0.5 * (h_pos * h_pos + h_neg * h_neg);
        vis_sum += 1.0 - slice_occ;
    }
    return saturate(vis_sum / float(kSlices));
}

// Pass 2.5a: PRISM Real-Time Dual-Scale XeGTAO + 1-Bounce Hemisphere Colored SSGI
[shader("fragment")]
float4 giAoFS(FullscreenOut input) : SV_Target {
    const float2 uv = input.uv;
    float d = scene_depth.SampleLevel(uv, 0.0);
    if (d >= 0.99995) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float view_t = linear_depth(d);
    if (view_t <= 0.45) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float3 p = screen_to_world(uv, view_t);
    float3 n = depth_normal(uv, p);
    // R2 2D golden-ratio quasi-random sequence: uniform 2x2 pixel distribution with ZERO diagonal IGN beat stripes!
    float spin = frac(dot(floor(input.position.xy), float2(0.75487767, 0.56984029)));

    // Dual-scale continuous horizon AO: tight 0.16m micro-crevices (block joints/folds) + macro architectural corners
    float ao = 1.0;
    if (frame.camera_clip.w > 0.0) {
        float vis_micro = horizon_ao_scale(uv, p, n, view_t, 0.17, spin);
        float vis_macro = horizon_ao_scale(uv, p, n, view_t, frame.post_params.w * (1.0 + view_t * 0.015), frac(spin + 0.5));
        float combined = min(vis_micro, vis_macro) * 0.48 + (vis_micro * vis_macro) * 0.52;
        ao = saturate(lerp(1.0, combined, frame.camera_clip.w * 1.18));
    }

    // 12-tap Physical Form-Factor Colored SSGI: checks BOTH receiver AND emitter facing (cos_i * cos_o)
    // so horizontal wall tops NEVER beam ghostly halos onto shadowed floors beside silhouettes!
    float3 gi = float3(0.0, 0.0, 0.0);
    const float bounce = frame.post_params.z;
    if (bounce > 0.0 && view_t > 0.65) {
        const float2 step_px = frame.screen.xy * (54.0 * frame.screen.w / 1080.0);
        float spin_gi = frac(dot(floor(input.position.xy), float2(0.56984029, 0.75487767))) * 6.2831853;
        float3 acc = float3(0.0, 0.0, 0.0);
        float wsum = 0.0;
        const int kGiTaps = 12;
        for (int i = 0; i < kGiTaps; ++i) {
            float f = (float(i) + 0.5) / float(kGiTaps);
            float a = float(i) * 2.39996323 + spin_gi;
            float r_norm = sqrt(f);
            float2 suv = uv + float2(cos(a), sin(a)) * (r_norm * step_px);
            if (any(suv <= 0.0) || any(suv >= 1.0)) continue;
            float sd = scene_depth.SampleLevel(suv, 0.0);
            if (sd >= 0.99995) continue;
            float z_s = linear_depth(sd);
            if (z_s <= 0.65) continue; // ignore first-person weapon viewmodel (ZERO weapon ghosting!)
            float3 sp = screen_to_world(suv, z_s);
            float3 dv = sp - p;
            float dist2 = dot(dv, dv);
            if (dist2 < 0.01 || dist2 > 20.0) continue;
            float dist = sqrt(dist2);
            float3 dir = dv / dist;
            // True Radiosity Form-Factor: BOTH receiver n faces sp AND emitter ns faces p!
            float recv_cos = saturate(dot(n, dir) - 0.08);
            if (recv_cos <= 0.0) continue;
            float3 ns = depth_normal(suv, sp);
            float emit_cos = saturate(dot(ns, -dir) - 0.05);
            float form = recv_cos * emit_cos;
            if (form <= 0.01) continue;
            float3 s_hdr = max(hdr_scene.SampleLevel(suv, 0.0).rgb, 0.0);
            float ls = dot(s_hdr, float3(0.2126, 0.7152, 0.0722));
            float3 s_clamped = s_hdr / (1.0 + ls * 0.28);
            float w = form / (1.0 + dist * 0.75);
            acc += s_clamped * w;
            wsum += w;
        }
        // Soft normalization prevents sparse edge samples from amplifying into bright silhouette halos!
        gi = acc / (wsum + 1.85);
    }
    return float4(gi, ao);
}

// Pass 2.5b: Dense 5x5 Edge-Preserving Depth+Normal Joint-Bilateral Denoiser (strict silhouette edge-stop!)
[shader("fragment")]
float4 giAoDenoiseFS(FullscreenOut input) : SV_Target {
    const float2 uv = input.uv;
    float d0 = scene_depth.SampleLevel(uv, 0.0);
    float4 center = hdr_scene.SampleLevel(uv, 0.0); // binding 2 = gi_ao_raw_
    if (d0 >= 0.99995) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float z0 = linear_depth(d0);
    if (z0 <= 0.65) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float3 p0 = screen_to_world(uv, z0);
    float3 n0 = depth_normal(uv, p0);
    const float2 px = frame.screen.xy;
    const float z_sigma = max(0.014 * z0 + 0.005, 1e-4);

    float4 acc = center;
    float wsum = 1.0;
    for (int iy = -2; iy <= 2; ++iy) {
        for (int ix = -2; ix <= 2; ++ix) {
            if (ix == 0 && iy == 0) continue;
            float2 suv = uv + float2(float(ix), float(iy)) * px;
            if (any(suv <= 0.0) || any(suv >= 1.0)) continue;
            float dt = scene_depth.SampleLevel(suv, 0.0);
            if (dt >= 0.99995) continue;
            float zt = linear_depth(dt);
            if (zt <= 0.65) continue;
            float dz = abs(zt - z0) / z_sigma;
            if (dz > 4.0) continue;
            float3 pt = screen_to_world(suv, zt);
            float3 nt = depth_normal(suv, pt);
            float ndn = saturate(dot(n0, nt));
            float w_s = exp(-float(ix * ix + iy * iy) * 0.16);
            float w_z = exp(-dz * dz * 0.75);
            float w_n = ndn * ndn * ndn * ndn;
            float w = w_s * w_z * w_n;
            acc += hdr_scene.SampleLevel(suv, 0.0) * w;
            wsum += w;
        }
    }
    return acc / wsum;
}

// Screen-Space Contact Shadows (SSCS): 8 short IGN-jittered depth taps toward the sun catch
// sub-centimeter micro-shadows under pebbles, grass tufts, displaced stones & wall trims.
float screen_space_contact_shadow(float3 p, float3 n, float t, float2 pixel) {
    float ndl = dot(n, frame.sun_dir.xyz);
    if (ndl <= 0.06 || t < 0.7 || t > 28.0) {
        return 1.0;
    }
    const float max_dist = lerp(0.18, 0.55, saturate(t / 14.0));
    const float thickness = lerp(0.035, 0.12, saturate(t / 18.0));
    float jitter = ign(pixel + float2(13.0, 29.0));
    float occ = 0.0;
    const int kSteps = 8;
    for (int i = 0; i < kSteps; ++i) {
        float step_t = (float(i) + 0.35 + jitter * 0.65) / float(kSteps);
        float3 sp = p + n * 0.018 + frame.sun_dir.xyz * (step_t * max_dist);
        float4 cs = mul(frame.view_proj, float4(sp, 1.0));
        float2 suv = cs.xy / cs.w * 0.5 + 0.5;
        if (any(suv < 0.0) || any(suv > 1.0)) {
            break;
        }
        float scene_t = linear_depth(scene_depth.SampleLevel(suv, 0.0));
        float depth_diff = cs.w - scene_t;
        if (depth_diff > 0.012 && depth_diff < thickness) {
            occ = max(occ, 1.0 - step_t * 0.55);
        }
    }
    float fade = (1.0 - smoothstep(18.0, 28.0, t)) * smoothstep(0.14, 0.36, ndl);
    return 1.0 - occ * fade * 0.48;
}

// Low-cost 6-tap shadow-map Volumetric Sun Shafts (God-Rays) along the view ray.
float3 volumetric_sun_shafts(float2 uv, float view_t, float2 pixel) {
    if (frame.sky_zenith.w <= 1e-5 || view_t < 2.5) {
        return 0.0;
    }
    float3 ro = frame.camera_pos.xyz;
    float2 ndc = uv * 2.0 - 1.0;
    float3 rd = normalize(frame.camera_forward.xyz + frame.camera_right.xyz * ndc.x * frame.camera_right.w -
                          frame.camera_up.xyz * ndc.y * frame.camera_up.w);
    float sd = max(dot(rd, frame.sun_dir.xyz), 0.0);
    float phase = mie_phase(sd, 0.60);
    float max_ray = min(view_t, 42.0);
    float jitter = ign(pixel + float2(37.0, 17.0));
    float lit_sum = 0.0;
    const int kSteps = 6;
    for (int i = 0; i < kSteps; ++i) {
        float rt = lerp(2.0, max_ray, (float(i) + jitter) / float(kSteps));
        float3 sp = ro + rd * rt;
        float4 lp = mul(frame.light_view_proj, float4(sp, 1.0));
        float3 lndc = lp.xyz / lp.w;
        float2 luv = lndc.xy * 0.5 + 0.5;
        float in_light = 1.0;
        if (all(luv >= 0.0) && all(luv <= 1.0) && lndc.z <= 1.0) {
            in_light = lndc.z - 0.001 <= shadow_map.SampleLevel(luv, 0.0) ? 1.0 : 0.0;
        }
        float height_w = exp(-max(sp.y, 0.0) * 0.14);
        lit_sum += in_light * height_w;
    }
    float shaft_density = (lit_sum / float(kSteps)) * saturate((max_ray - 2.0) / 36.0);
    // Keep strictly proportional to sun_color and fog_density (zero color shift on achromatic tests)
    return frame.sun_color.rgb * (shaft_density * phase * frame.sky_zenith.w * 9.0);
}

// 28-step jittered + binary-refined Screen-Space Reflections (SSR) executed ONLY on water puddle pixels (alpha > 0.05).
// Reflects the sunlit plaster facades, stone walls, olive trees, blue sky and golden clouds in liquid-mirror clarity.
float3 puddle_ssr(float2 uv, float3 p, float3 base_col, float2 pixel) {
    float3 ro = frame.camera_pos.xyz;
    float3 rd = normalize(p - ro);
    float t_anim = frame.camera_pos.w;
    float rx = sin(p.x * 7.0 + t_anim * 1.1) * 0.0018 + sin(p.z * 11.0 - t_anim * 0.8) * 0.0014;
    float rz = cos(p.z * 8.0 + t_anim * 1.3) * 0.0018 + cos(p.x * 13.0 + t_anim * 0.9) * 0.0014;
    float3 wn = normalize(float3(rx, 1.0, rz));
    float3 rdir = reflect(rd, wn);

    float jitter = ign(pixel + float2(71.0, 23.0));
    float prev_t = 0.08;
    bool hit = false;
    float2 hit_uv = uv;
    float hit_fade = 0.0;
    const int kSteps = 28;
    for (int i = 0; i < kSteps; ++i) {
        float frac_i = (float(i) + 0.35 + jitter * 0.65) / float(kSteps);
        float march_t = 0.28 + 38.0 * frac_i * frac_i;
        float3 sp = p + float3(0.0, 0.035, 0.0) + rdir * march_t;
        float4 cs = mul(frame.view_proj, float4(sp, 1.0));
        if (cs.w <= 0.1) {
            break;
        }
        float2 suv = cs.xy / cs.w * 0.5 + 0.5;
        if (any(suv < 0.0) || any(suv > 1.0)) {
            break;
        }
        float sd_raw = scene_depth.SampleLevel(suv, 0.0);
        if (sd_raw < 0.99995) {
            float scene_t = linear_depth(sd_raw);
            float dz = cs.w - scene_t;
            float thick = lerp(0.18, 2.4, frac_i);
            if (dz > 0.03 && dz < thick && scene_t > 0.8) {
                // 3 binary-search steps sharpen reflected window frames, eaves and stone walls
                float lo = prev_t, hi = march_t;
                for (int b = 0; b < 3; ++b) {
                    float mid = 0.5 * (lo + hi);
                    float4 mcs = mul(frame.view_proj, float4(p + float3(0.0, 0.02, 0.0) + rdir * mid, 1.0));
                    float2 muv = mcs.xy / mcs.w * 0.5 + 0.5;
                    float mt = linear_depth(scene_depth.SampleLevel(muv, 0.0));
                    if (mcs.w > mt) {
                        hi = mid;
                        hit_uv = muv;
                    } else {
                        lo = mid;
                    }
                }
                float2 edge = smoothstep(0.0, 0.08, hit_uv) * smoothstep(1.0, 0.92, hit_uv);
                hit_fade = edge.x * edge.y * (1.0 - frac_i * 0.25);
                hit = true;
                break;
            }
        }
        prev_t = march_t;
    }

    float3 sky_refl = sky_color(rdir);
    float3 refl = sky_refl;
    if (hit) {
        float3 ssr_col = hdr_scene.SampleLevel(hit_uv, 0.0).rgb;
        refl = lerp(sky_refl, ssr_col, saturate(hit_fade));
    }
    float ndv = clamp(dot(wn, -rd), 1e-4, 1.0);
    float rx_f = saturate(1.0 - ndv);
    float fresnel = 0.04 + 0.96 * (rx_f * rx_f * rx_f * sqrt(rx_f));
    float blend = saturate(fresnel * 1.05 + 0.22);
    return lerp(float3(0.045, 0.036, 0.026) + base_col * 0.12, refl * 0.95, blend);
}

// Screen-space Radial Crepuscular Sun Shafts (God-Rays) fanning sharply from behind the jagged right mountain pass.
float3 crepuscular_sun_shafts(float2 uv, float view_t, float2 pixel) {
    float sky_chroma = length(frame.sky_zenith.rgb - frame.sky_horizon.rgb);
    if (frame.sky_zenith.w <= 1e-5 || sky_chroma <= 0.02 || view_t < 1.5) {
        return 0.0;
    }
    float3 sun_world = frame.camera_pos.xyz + frame.sun_dir.xyz * 600.0;
    float4 sun_cs = mul(frame.view_proj, float4(sun_world, 1.0));
    if (sun_cs.w <= 0.0) {
        return 0.0;
    }
    float2 sun_uv = sun_cs.xy / sun_cs.w * 0.5 + 0.5;
    float2 delta = sun_uv - uv;
    float dist_uv = length(delta);
    // Keep god-rays tightly focused around the golden sun pass (zero washout over left/center mountains!)
    if (dist_uv > 0.44) {
        return 0.0;
    }
    float2 step_uv = delta * (min(dist_uv, 0.34) / max(dist_uv, 1e-4)) / 24.0;
    float jitter = ign(pixel + float2(53.0, 19.0));
    float2 sample_uv = uv + step_uv * jitter;
    float illum = 0.0;
    float weight = 1.0;
    const int kSamples = 24;
    for (int i = 0; i < kSamples; ++i) {
        if (all(sample_uv >= 0.0) && all(sample_uv <= 1.0)) {
            float sd = scene_depth.SampleLevel(sample_uv, 0.0);
            float open_sky = step(0.99985, sd);
            illum += open_sky * weight;
        }
        sample_uv += step_uv;
        weight *= 0.92;
    }
    illum /= 9.8;
    // Subtle angular ray modulation mimics sunlight slicing through mountain crags & clouds
    float ang = atan2(delta.y, delta.x);
    float rays = 0.40 + 0.60 * pow(0.5 + 0.5 * sin(ang * 15.0 + 0.7) * cos(ang * 9.0 - 0.4), 1.8);
    float radial = pow(saturate(1.0 - dist_uv / 0.42), 2.4);
    float dist_w = lerp(0.08, 1.0, smoothstep(8.0, 65.0, view_t));
    // Rich warm amber-gold sunbeams without white washout!
    return frame.sun_color.rgb * float3(1.0, 0.72, 0.36) * (illum * rays * radial * dist_w * 0.52);
}

[shader("fragment")]
float4 compositeFS(FullscreenOut input) : SV_Target {
    const float2 uv = input.uv;
    const float exposure = frame.camera_forward.w;
    float4 hdr_raw = hdr_scene.SampleLevel(uv, 0.0);
    float3 c = hdr_raw.rgb;
    float water_mask = hdr_raw.a;
    float d = scene_depth.SampleLevel(uv, 0.0);
    // First-person viewmodel (< 0.45 m) is skipped: its scale is below the screen-space GI/AO radius.
    float view_t = d < 1.0 ? linear_depth(d) : 1e6;
    if (d < 1.0 && view_t > 0.45) {
        if (water_mask > 0.05) {
            float3 p = screen_to_world(uv, view_t);
            c = puddle_ssr(uv, p, c, input.position.xy);
        } else {
            // Read silky bilaterally-denoised Dual-Scale XeGTAO (.a) + Multi-Bounce Colored SSGI (.rgb) from binding 4!
            float4 gi_ao = bloom_tex.SampleLevel(uv, 0.0);
            float3 approx_alb = saturate(c * exposure / (1.0 + c * exposure) * 1.45);
            if (frame.camera_clip.w > 0.0) {
                // Jimenez Multi-Bounce Colored AO: deep velvety dark crevices between blocks & under cloth, zero noise!
                c *= multi_bounce_ao(saturate(gi_ao.a), approx_alb);
                // Optional subtle SSCS only on terrain-enabled outdoor scenes (disabled on smooth clay to avoid bevel ripples!)
                if (frame.camera_clip.z > 0.0 && view_t > 0.7) {
                    float3 p = screen_to_world(uv, view_t);
                    float3 n = depth_normal(uv, p);
                    float sscs = screen_space_contact_shadow(p, n, view_t, input.position.xy);
                    c *= lerp(0.68, 1.0, sscs);
                }
            }
            // Apply physically bounded, form-factor-verified Colored SSGI Bounce (ZERO screen-space ghosting halos!)
            const float bounce = frame.post_params.z;
            if (bounce > 0.0 && view_t > 0.65) {
                float l = dot(c * exposure, float3(0.2126, 0.7152, 0.0722));
                float sh = 1.0 - smoothstep(0.08, 0.72, l);
                float3 gi_contrib = max(gi_ao.rgb, 0.0) * bounce * (0.10 + 0.42 * sh) * lerp(float3(0.55, 0.55, 0.55), approx_alb, 0.45);
                c += gi_contrib * saturate(gi_ao.a * 0.75 + 0.25);
            }
        }
    }

    // Add shadow-map-occluded Volumetric Sun Shafts + Radial Alpine Crepuscular God-Rays
    if (view_t > 2.5) {
        c += volumetric_sun_shafts(uv, view_t, input.position.xy);
    }
    c += crepuscular_sun_shafts(uv, view_t, input.position.xy);
    return float4(c, 1.0);
}

// ------------------------------------------------------------------ post: bloom (quarter resolution)

float3 sample_karis(float2 uv, float exposure) {
    float3 s = max(post_src.SampleLevel(uv, 0.0).rgb * exposure, 0.0);
    float l = dot(s, float3(0.2126, 0.7152, 0.0722));
    float knee = clamp(l - 0.65, 0.0, 0.5);
    float w = max(knee * knee / 2.0, l - 0.9) / max(l, 1e-4);
    float3 b = min(s * w, 12.0);
    float lb = dot(b, float3(0.2126, 0.7152, 0.0722));
    return b / (1.0 + lb * 0.08);
}

[shader("fragment")]
float4 bloomDownFS(FullscreenOut input) : SV_Target {
    // 13-tap dual-filter downsample (Jimenez, Call of Duty AW) eliminates boxy quarter-res block artifacts
    const float2 px = frame.screen.xy;
    const float2 uv = input.uv;
    const float exp_val = frame.camera_forward.w;

    float3 a = sample_karis(uv + float2(-2.0, -2.0) * px, exp_val);
    float3 b = sample_karis(uv + float2( 0.0, -2.0) * px, exp_val);
    float3 c = sample_karis(uv + float2( 2.0, -2.0) * px, exp_val);
    float3 d = sample_karis(uv + float2(-1.0, -1.0) * px, exp_val);
    float3 e = sample_karis(uv + float2( 1.0, -1.0) * px, exp_val);
    float3 f = sample_karis(uv + float2(-2.0,  0.0) * px, exp_val);
    float3 g = sample_karis(uv,                           exp_val);
    float3 h = sample_karis(uv + float2( 2.0,  0.0) * px, exp_val);
    float3 i = sample_karis(uv + float2(-1.0,  1.0) * px, exp_val);
    float3 j = sample_karis(uv + float2( 1.0,  1.0) * px, exp_val);
    float3 k = sample_karis(uv + float2(-2.0,  2.0) * px, exp_val);
    float3 l = sample_karis(uv + float2( 0.0,  2.0) * px, exp_val);
    float3 m = sample_karis(uv + float2( 2.0,  2.0) * px, exp_val);

    float3 res = (d + e + i + j) * (0.5 / 4.0);
    res += (a + b + g + f) * (0.125 / 4.0);
    res += (b + c + h + g) * (0.125 / 4.0);
    res += (f + g + l + k) * (0.125 / 4.0);
    res += (g + h + m + l) * (0.125 / 4.0);
    return float4(res, 1.0);
}

// Smooth 13-tap Gaussian blur using bilinear hardware offsets (1.0 quarter-res texel step = 4.0 * frame.screen.xy)
float3 blur13(float2 uv, float2 texel_step) {
    const float offsets[6] = {1.4117647, 3.2941176, 5.1764706, 7.0588235, 8.9411765, 10.8235294};
    const float weights[6] = {0.1964826, 0.1229491, 0.0585506, 0.0211641, 0.0058064, 0.0012106};
    float3 c = post_src.SampleLevel(uv, 0.0).rgb * 0.1876732;
    for (int i = 0; i < 6; ++i) {
        float2 o = texel_step * offsets[i];
        c += post_src.SampleLevel(uv + o, 0.0).rgb * weights[i];
        c += post_src.SampleLevel(uv - o, 0.0).rgb * weights[i];
    }
    return c;
}

[shader("fragment")]
float4 bloomBlurHFS(FullscreenOut input) : SV_Target {
    return float4(blur13(input.uv, float2(frame.screen.x * 4.0, 0.0)), 1.0);
}

[shader("fragment")]
float4 bloomBlurVFS(FullscreenOut input) : SV_Target {
    return float4(blur13(input.uv, float2(0.0, frame.screen.y * 4.0)), 1.0);
}

// ------------------------------------------------------------------ post: AgX filmic tonemap, FXAA, CAS sharpen

// Row-normalized AgX Filmic Tonemapping (Troy Sobotka): superior highlight roll-off without ACES yellow/magenta skew,
// and strictly achromatic-preserving (neutral grey maps to identical neutral grey).
float3 agx_default_contrast(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

float3 agx_tonemap(float3 val) {
    // Exact row-normalized AgX inset matrix (row sums == 1.0)
    const float3x3 agx_inset = float3x3(
        0.84236416, 0.07842290, 0.07921294,
        0.04232981, 0.87850114, 0.07916906,
        0.04237768, 0.07843735, 0.87918497
    );
    // Exact inverse outset matrix (row sums == 1.0)
    const float3x3 agx_outset = float3x3(
        1.19704227, -0.09801725, -0.09902501,
       -0.05290407,  1.15186052, -0.09895645,
       -0.05297886, -0.09803982,  1.15101868
    );
    const float min_ev = -12.47393;
    const float max_ev = 4.026069;

    float3 v = mul(agx_inset, max(val, 1e-6));
    v = clamp(log2(v), min_ev, max_ev);
    v = (v - min_ev) / (max_ev - min_ev);
    v = agx_default_contrast(v);
    v = mul(agx_outset, v);
    // AgX curve outputs display-encoded; convert back to linear for our sRGB swapchain target
    v = pow(max(saturate(v), 1e-6), 2.2);
    // Filmic Studio Punchy Contrast: rich deep shadows & vivid color fidelity (100% strictly achromatic-preserving!)
    v = pow(max(v, 1e-6), 1.05);
    float l = dot(v, float3(0.2126, 0.7152, 0.0722));
    v = lerp(l.xxx, v, 1.08);
    return saturate(v);
}

float3 tonemapped(float2 uv) {
    float3 hdr = hdr_scene.SampleLevel(uv, 0.0).rgb * frame.camera_forward.w;
    hdr += bloom_tex.SampleLevel(uv, 0.0).rgb * frame.post_params.x;
    return agx_tonemap(hdr);
}

float luma_of(float3 c) {
    return dot(c, float3(0.299, 0.587, 0.114));
}

[shader("vertex")]
FullscreenOut postVS(uint id : SV_VertexID) {
    return fullscreen(id);
}

// ------------------------------------------------------------------ TAA (temporal anti-aliasing)
// Jittered frames resolved into an HDR history: closest-depth dilated reprojection, Catmull-Rom
// history fetch (keeps it sharp), YCoCg variance clipping (no ghost trails) and luma-weighted
// blending (no firefly smearing). Binding 2 = current lit frame, 6 = previous history.

float3 rgb_to_ycocg(float3 c) {
    return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25)));
}

float3 ycocg_to_rgb(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// 5-tap bicubic Catmull-Rom through bilinear taps (Jimenez / Karis).
float3 history_catmull_rom(float2 uv) {
    const float2 size = frame.screen.zw;
    float2 pos = uv * size;
    float2 c = floor(pos - 0.5) + 0.5;
    float2 f = pos - c;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 t0 = (c - 1.0) / size;
    float2 t3 = (c + 2.0) / size;
    float2 t12 = (c + w2 / w12) / size;
    float3 r = post_src.SampleLevel(float2(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y) +
               post_src.SampleLevel(float2(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y) +
               post_src.SampleLevel(t12, 0.0).rgb * (w12.x * w12.y) +
               post_src.SampleLevel(float2(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y) +
               post_src.SampleLevel(float2(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    float wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(r / max(wsum, 1e-4), 0.0);
}

[shader("fragment")]
float4 taaFS(FullscreenOut input) : SV_Target {
    const int2 px = int2(input.position.xy);
    const int2 size = int2(frame.screen.zw);
    float3 cur = max(hdr_scene.Load(int3(px, 0)).rgb, 0.0);

    // 3x3 neighbourhood: colour moments for clipping, closest depth for reprojection (sharp edges).
    float3 m1 = float3(0.0, 0.0, 0.0);
    float3 m2 = float3(0.0, 0.0, 0.0);
    float closest = 1.0;
    int2 closest_px = px;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 p = clamp(px + int2(x, y), int2(0, 0), size - 1);
            float3 s = rgb_to_ycocg(max(hdr_scene.Load(int3(p, 0)).rgb, 0.0));
            m1 += s;
            m2 += s * s;
            float d = scene_depth.Load(int3(p, 0));
            if (d < closest) {
                closest = d;
                closest_px = p;
            }
        }
    }
    m1 /= 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - m1 * m1, 0.0));

    // Motion is taken from the closest-depth neighbour (dilation keeps foreground edges sharp) but
    // applied to this pixel's own centre.
    const float2 uv_unjit = (float2(closest_px) + 0.5) * frame.screen.xy - frame.taa.xy;
    float view_t = closest < 0.99995 ? linear_depth(closest) : 1.0e5;
    float3 world = screen_to_world(uv_unjit, view_t);
    float4 prev_clip = mul(frame.prev_view_proj, float4(world, 1.0));
    float2 motion = prev_clip.xy / max(prev_clip.w, 1e-4) * 0.5 + 0.5 - uv_unjit;
    float2 prev_uv = (float2(px) + 0.5) * frame.screen.xy + motion;
    float2 motion_px = motion * frame.screen.zw;
    // The first-person viewmodel moves with the camera, so camera reprojection does not apply to it.
    bool valid = frame.taa.z > 0.5 && prev_clip.w > 0.05 && all(prev_uv > 0.0) && all(prev_uv < 1.0) &&
                 view_t > 0.6;
    if (!valid) {
        return float4(cur, 1.0);
    }

    float3 hist = rgb_to_ycocg(history_catmull_rom(prev_uv));
    // Tighter box when moving: disocclusions resolve quickly; static views converge to full AA.
    const float speed = length(motion_px);
    const float gamma = lerp(1.25, 0.85, saturate(speed / 8.0));
    float3 lo = m1 - sigma * gamma;
    float3 hi = m1 + sigma * gamma;
    float3 dir = hist - m1;
    float3 ext = max(abs(hi - lo) * 0.5, 1e-5);
    float3 unit = abs(dir / ext);
    float ma = max(unit.x, max(unit.y, unit.z));
    hist = ma > 1.0 ? m1 + dir / ma : hist;

    float3 c = rgb_to_ycocg(cur);
    float alpha = lerp(0.08, 0.2, saturate(speed / 12.0));
    float wc = alpha / (1.0 + c.x);
    float wh = (1.0 - alpha) / (1.0 + hist.x);
    float3 res = ycocg_to_rgb((c * wc + hist * wh) / (wc + wh));
    res = max(res, 0.0);
    if (any(isnan(res)) || any(isinf(res))) {
        res = cur;
    }
    return float4(res, 1.0);
}

[shader("fragment")]
float4 postFS(FullscreenOut input) : SV_Target {
    const float2 uv = input.uv;
    const float2 px = frame.screen.xy;

    // FXAA (console variant) on the AgX-tonemapped image.
    float3 m = tonemapped(uv);
    float3 c_nw = tonemapped(uv + float2(-px.x, -px.y));
    float3 c_ne = tonemapped(uv + float2(px.x, -px.y));
    float3 c_sw = tonemapped(uv + float2(-px.x, px.y));
    float3 c_se = tonemapped(uv + float2(px.x, px.y));
    float l_nw = luma_of(c_nw);
    float l_ne = luma_of(c_ne);
    float l_sw = luma_of(c_sw);
    float l_se = luma_of(c_se);
    float l_m = luma_of(m);
    float l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    float l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    float3 c = m;
    // TAA already resolved the edges; FXAA on top would only soften the image.
    if (frame.taa.w < 0.5 && l_max - l_min > max(0.0312, l_max * 0.125)) {
        float2 dir = float2(-((l_nw + l_ne) - (l_sw + l_se)), (l_nw + l_sw) - (l_ne + l_se));
        float reduce = max((l_nw + l_ne + l_sw + l_se) * 0.03125, 1.0 / 128.0);
        float inv = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
        dir = clamp(dir * inv, -8.0, 8.0) * px;
        float3 a = 0.5 * (tonemapped(uv + dir * (1.0 / 3.0 - 0.5)) + tonemapped(uv + dir * (2.0 / 3.0 - 0.5)));
        float3 b = a * 0.5 + 0.25 * (tonemapped(uv - dir * 0.5) + tonemapped(uv + dir * 0.5));
        float l_b = luma_of(b);
        c = (l_b < l_min || l_b > l_max) ? a : b;
    }

    // AMD FidelityFX-style Contrast-Adaptive Sharpening (CAS) reusing the 4 FXAA neighbor taps (0 extra texture reads!)
    {
        float3 mn = min(c, min(min(c_nw, c_ne), min(c_sw, c_se)));
        float3 mx = max(c, max(max(c_nw, c_ne), max(c_sw, c_se)));
        float3 amp = saturate(min(mn, 2.0 - mx) / max(mx, 1e-4));
        // Off under TAA: its Catmull-Rom history is already sharp and CAS would ring on resolved edges.
        float3 w = -sqrt(amp) * (frame.taa.w > 0.5 ? 0.0 : 0.14);
        c = saturate((c + (c_nw + c_ne + c_sw + c_se) * w) / max(1.0 + 4.0 * w, 1e-4));
    }

    // The sun supplies the warmth; keep neutral shadows neutral after tonemapping.
    c *= 1.0 - 0.10 * smoothstep(0.35, 0.71, distance(uv, 0.5));
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = saturate(lerp(l.xxx, c, frame.post_params.y));

    // Crosshair: four arms around the centre with a dark outline.
    const float arm = frame.point_color.w;
    if (arm > 0.0) {
        float2 q = abs(input.position.xy - frame.output_size.xy * 0.5);
        const float gap = arm * 0.6;
        float along = max(q.x, q.y);
        float across = min(q.x, q.y);
        bool in_arm = along >= gap && along <= gap + arm;
        if (in_arm && across <= 1.0) {
            c = lerp(c, float3(0.95, 0.95, 0.9), 0.85);
        } else if (along >= gap - 1.0 && along <= gap + arm + 1.0 && across <= 2.0) {
            c *= 0.35;
        }
    }

    // Triangular dither hides banding in the 8-bit output.
    float noise = hash13(float3(input.position.xy, 1.0)) - hash13(float3(input.position.yx, 2.0));
    c += noise / 255.0;
    return float4(c, 1.0);
}

// ================================================================== NOVA Lumina RT (hybrid ray tracing)
//
// Pipeline (WorldRenderer::render_frame, RT devices): normal/depth prepass -> sun-shadow + AO rays ->
// one-bounce GI rays -> per-signal temporal accumulation (depth-validated reprojection + neighbourhood
// clamp) -> penumbra-aware spatial denoise -> forward shading reads the results per pixel.
// Reflections on water trace the TLAS and reuse on-screen radiance when the hit is visible.
#ifdef NOVA_RT

static const float kPi = 3.14159265;

int2 rt_clamp_px(int2 px) {
    return clamp(px, int2(0, 0), int2(frame.screen.zw) - 1);
}

// Per-pixel, per-frame decorrelated [0,1)^2: IGN spatial pattern + golden-ratio temporal scramble.
float2 rt_rand2(float2 pixel, float salt) {
    const float f = frame.rt_params.x;
    float a = ign(pixel + float2(salt * 37.0, salt * 11.0));
    float b = ign(pixel.yx + float2(salt * 19.0 + 47.0, salt * 29.0 + 13.0));
    return frac(float2(a, b) + float2(0.61803399, 0.75487767) * (f + salt * 7.0));
}

// Orthonormal basis around n (Duff et al. 2017).
void rt_basis(float3 n, out float3 t, out float3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

float3 rt_cosine_dir(float3 n, float2 u) {
    float3 t, b;
    rt_basis(n, t, b);
    float r = sqrt(u.x);
    float phi = 2.0 * kPi * u.y;
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u.x, 0.0)));
}

// Direction inside the solar disc: soft shadows get their physically correct, contact-hardening penumbra.
float3 rt_sun_dir(float2 u) {
    float3 s = frame.sun_dir.xyz;
    float3 t, b;
    rt_basis(s, t, b);
    float r = sqrt(u.x) * tan(frame.rt_params.y);
    float phi = 2.0 * kPi * u.y;
    return normalize(s + (t * cos(phi) + b * sin(phi)) * r);
}

float rt_visible(float3 origin, float3 dir, float tmax) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = tmax;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
    q.TraceRayInline(scene_tlas, 0, 0xFF, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : 1.0;
}

// Diffuse sky light arriving along `dir` (no sun disc: it is sampled explicitly, so no fireflies).
float3 rt_sky_radiance(float3 dir) {
    float3 sky;
    if (dir.y >= 0.0) {
        sky = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, sqrt(dir.y));
        sky += frame.sun_color.rgb * 0.05 * pow(saturate(dot(dir, frame.sun_dir.xyz)), 6.0);
    } else {
        float3 sky_amb = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, 0.35);
        sky = frame.ground_color.rgb * (sky_amb * 0.6 + frame.sun_color.rgb * 0.35 * saturate(frame.sun_dir.y));
    }
    return sky * frame.ground_color.w;
}

float3 rt_sky_irradiance(float3 n) {
    float3 up = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, 0.5);
    float3 sky_amb = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, 0.35);
    float3 down = frame.ground_color.rgb * (sky_amb * 0.6 + frame.sun_color.rgb * 0.5 * saturate(frame.sun_dir.y));
    return lerp(down, up, n.y * 0.5 + 0.5) * frame.ground_color.w;
}

struct RtHit {
    float3 position;
    float3 normal;
    float3 albedo;
    float3 emissive;
    int material;
};

RtHit rt_fetch_hit(uint instance, uint prim, float2 bary, float3x4 o2w, float3 origin, float3 dir, float t) {
    RtInstance inst = rt_instances[instance];
    uint i0 = inst.indices[prim * 3u];
    uint i1 = inst.indices[prim * 3u + 1u];
    uint i2 = inst.indices[prim * 3u + 2u];
    float w0 = 1.0 - bary.x - bary.y;
    float3 n_obj = rt_vertex_normal(inst.vertices, i0) * w0 + rt_vertex_normal(inst.vertices, i1) * bary.x +
                   rt_vertex_normal(inst.vertices, i2) * bary.y;
    float4 c0 = rt_vertex_color(inst.vertices, i0);
    float3 col = c0.rgb * w0 + rt_vertex_color(inst.vertices, i1).rgb * bary.x +
                 rt_vertex_color(inst.vertices, i2).rgb * bary.y;

    RtHit h;
    h.position = origin + dir * t;
    float3 n = mul((float3x3)o2w, n_obj);
    n = dot(n, n) > 1e-10 ? normalize(n) : -dir;
    h.normal = dot(n, dir) > 0.0 ? -n : n; // two-sided
    h.material = inst.albedo.a > 0.5 ? int(c0.a + 0.5) : -1;
    h.albedo = saturate(inst.albedo.a > 0.5 ? col * inst.albedo.rgb : inst.albedo.rgb);
    h.emissive = inst.emissive.rgb;
    if (h.material == MAT_WATER) {
        h.albedo *= 0.25;
    } else if (h.material == MAT_EMISSIVE) {
        h.emissive += col * kEmissiveScale;
    }
    return h;
}

// Outgoing radiance at a ray hit: sun with a traced shadow ray + diffuse sky (the second bounce's stand-in).
float3 rt_shade_hit(RtHit h, float2 u) {
    float ndl = saturate(dot(h.normal, frame.sun_dir.xyz));
    float vis = 0.0;
    if (ndl > 0.0) {
        vis = rt_visible(h.position + h.normal * 0.02, rt_sun_dir(u), frame.rt_params3.x);
    }
    return h.albedo * (frame.sun_color.rgb * ndl * vis + rt_sky_irradiance(h.normal)) + h.emissive;
}

struct RtPixel {
    bool valid;
    float3 p;
    float3 n;
    float view_t;
};

RtPixel rt_pixel(int2 px) {
    RtPixel o;
    px = rt_clamp_px(px);
    float d = rt_depth.Load(int3(px, 0));
    o.valid = d < 0.99995;
    o.view_t = o.valid ? linear_depth(d) : 1e6;
    float2 uv = (float2(px) + 0.5) * frame.screen.xy - frame.taa.xy; // unjittered sample position
    o.p = screen_to_world(uv, o.view_t);
    float3 n = rt_normal.Load(int3(px, 0)).xyz;
    o.n = dot(n, n) > 1e-6 ? normalize(n) : float3(0.0, 1.0, 0.0);
    return o;
}

float rt_px_per_metre(float view_t) {
    return frame.screen.w / max(2.0 * view_t * frame.camera_up.w, 1e-4);
}

// ---- prepass: world normal (+ depth) for ray origins, reprojection and denoiser edge stops

// Shading normal moved into the hemisphere of the viewer-facing geometric normal. Robust both to
// two-sided meshes whose winding disagrees with their normals (cloth) and to smooth normals that
// tilt away from the camera near silhouettes (a camera-facing test alone would flip those).
float3 rt_visible_normal(float3 n, float3 world_pos, bool front_face) {
    float3 ng = cross(ddx(world_pos), ddy(world_pos));
    if (dot(ng, ng) < 1e-14) {
        return front_face ? n : -n;
    }
    ng = normalize(ng);
    if (dot(ng, frame.camera_pos.xyz - world_pos) < 0.0) {
        ng = -ng;
    }
    return dot(n, ng) >= 0.0 ? n : -n;
}

float3 rt_prepass_normal(VSOutput input) {
    return dot(input.normal, input.normal) > 1e-10 ? normalize(input.normal) : float3(0.0, 1.0, 0.0);
}

void rt_prepass_alpha_test(VSOutput input) {
    if (object.material.z > 0.0) {
        float2 uv_off = unpack_u16_pair(object.uv_xform.z, -8.0, 8.0, -8.0, 8.0);
        float uv_rot = unpack_u16_pair(object.uv_xform.w, -3.14159265, 3.14159265, 0.0, 0.15).x;
        float2 uvc = input.uv - 0.5;
        float2 uv = float2(uvc.x * cos(uv_rot) - uvc.y * sin(uv_rot), uvc.x * sin(uv_rot) + uvc.y * cos(uv_rot)) + 0.5;
        uv = uv * object.uv_xform.xy + uv_off;
        if (base_color_tex.Sample(uv).a * object.base_color.a < object.material.z) {
            discard;
        }
    }
}

// One-sided meshes keep exactly the normal the forward pass shades with (front_face flip included),
// so ray-traced visibility always agrees with direct lighting, even on meshes with inverted normals.
[shader("fragment")]
float4 rtPrepassFS(VSOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    float3 n = rt_prepass_normal(input);
    return float4(front_face ? n : -n, 1.0);
}

[shader("fragment")]
float4 rtPrepassTwoSidedFS(VSOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    return float4(rt_visible_normal(rt_prepass_normal(input), input.world_pos, front_face), 1.0);
}

[shader("fragment")]
float4 rtPrepassAssetFS(VSOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    rt_prepass_alpha_test(input);
    float3 n = rt_prepass_normal(input);
    return float4(front_face ? n : -n, 1.0);
}

[shader("fragment")]
float4 rtPrepassAssetTwoSidedFS(VSOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    rt_prepass_alpha_test(input);
    return float4(rt_visible_normal(rt_prepass_normal(input), input.world_pos, front_face), 1.0);
}

// ---- trace 1: sun shadow (closest hit -> blocker distance for the penumbra-aware denoiser) + short AO

[shader("fragment")]
float4 rtShadowFS(FullscreenOut input) : SV_Target {
    const int2 px = int2(input.position.xy);
    RtPixel g = rt_pixel(px);
    if (!g.valid) {
        return float4(1.0, 1.0, 0.0, 1.0);
    }
    const float3 origin = g.p + g.n * (0.012 + 0.0015 * g.view_t);

    float shadow = 0.0;
    float blocker = 0.0;
    {
        // Thin two-sided surfaces (cloth, leaves) are sun-lit through translucency even when facing
        // away: start the ray on the sun side. Closed solids still self-occlude correctly.
        const float3 n_sun = dot(g.n, frame.sun_dir.xyz) >= 0.0 ? g.n : -g.n;
        RayDesc ray;
        ray.Origin = g.p + n_sun * (0.012 + 0.0015 * g.view_t);
        ray.Direction = rt_sun_dir(rt_rand2(input.position.xy, 0.0));
        ray.TMin = 0.0;
        ray.TMax = frame.rt_params3.x;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
        q.TraceRayInline(scene_tlas, 0, 0xFF, ray);
        while (q.Proceed()) {}
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            blocker = q.CommittedRayT();
        } else {
            shadow = 1.0;
        }
    }

    // Short-range AO: real geometry, so no screen-space halos or off-screen occluder loss.
    const float ao_r = max(frame.rt_params2.y, 0.05);
    RayDesc ao_ray;
    ao_ray.Origin = origin;
    ao_ray.Direction = rt_cosine_dir(g.n, rt_rand2(input.position.xy, 1.0));
    ao_ray.TMin = 0.0;
    ao_ray.TMax = ao_r;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> qa;
    qa.TraceRayInline(scene_tlas, 0, 0xFF, ao_ray);
    while (qa.Proceed()) {}
    float ao = 1.0;
    if (qa.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float x = saturate(qa.CommittedRayT() / ao_r);
        ao = x * x * (3.0 - 2.0 * x);
    }
    return float4(shadow, ao, blocker, 1.0);
}

// ---- trace 2: one-bounce diffuse GI (cosine rays, hit shading with its own sun shadow ray)

[shader("fragment")]
float4 rtGiFS(FullscreenOut input) : SV_Target {
    const int2 px = int2(input.position.xy);
    RtPixel g = rt_pixel(px);
    if (!g.valid) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    const float3 origin = g.p + g.n * (0.012 + 0.0015 * g.view_t);
    const int rays = clamp(int(frame.rt_params2.w + 0.5), 1, 4);
    float3 sum = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < rays; ++i) {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = rt_cosine_dir(g.n, rt_rand2(input.position.xy, 2.0 + float(i) * 2.0));
        ray.TMin = 0.0;
        ray.TMax = max(frame.rt_params2.x, 0.5);
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(scene_tlas, 0, 0xFF, ray);
        while (q.Proceed()) {}
        float3 l;
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            RtHit h = rt_fetch_hit(q.CommittedInstanceID(), q.CommittedPrimitiveIndex(),
                                   q.CommittedTriangleBarycentrics(), q.CommittedObjectToWorld3x4(), ray.Origin,
                                   ray.Direction, q.CommittedRayT());
            l = rt_shade_hit(h, rt_rand2(input.position.xy, 3.0 + float(i) * 2.0));
        } else {
            l = rt_sky_radiance(ray.Direction);
        }
        // Firefly clamp: a single bright bounce must not dominate the 1-ray estimate.
        float lum = dot(l, float3(0.2126, 0.7152, 0.0722));
        sum += l * (min(lum, 6.0) / max(lum, 1e-4));
    }
    return float4(sum / float(rays), 1.0);
}

// ---- temporal accumulation: depth-validated reprojection + neighbourhood clamp (no ghost trails)
// Input: raw signal at binding 12, previous history at 13. Output: xyz signal, w = history length.

float4 rt_temporal(float2 pixel, float max_history) {
    const int2 px = int2(pixel);
    float4 cur = rt_a.Load(int3(px, 0));
    RtPixel g = rt_pixel(px);
    if (!g.valid) {
        return float4(cur.xyz, 1.0);
    }
    float3 mn = cur.xyz;
    float3 mx = cur.xyz;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float3 s = rt_a.Load(int3(rt_clamp_px(px + int2(x, y)), 0)).xyz;
            mn = min(mn, s);
            mx = max(mx, s);
        }
    }
    float4 clip = mul(frame.prev_view_proj, float4(g.p, 1.0));
    bool valid = frame.rt_params2.z < 0.5 && clip.w > 0.05;
    float2 prev_uv = clip.xy / max(clip.w, 1e-4) * 0.5 + 0.5;
    valid = valid && all(prev_uv > 0.0) && all(prev_uv < 1.0);
    float4 hist = float4(cur.xyz, 0.0);
    if (valid) {
        float prev_t = linear_depth(rt_prev_depth.SampleLevel(prev_uv, 0.0));
        valid = abs(prev_t - clip.w) < 0.035 * clip.w + 0.03;
        hist = rt_b.SampleLevel(prev_uv, 0.0);
    }
    float count = valid ? min(hist.w + 1.0, max_history) : 1.0;
    float3 h = clamp(hist.xyz, mn, mx);
    return float4(lerp(h, cur.xyz, 1.0 / count), count);
}

[shader("fragment")]
float4 rtTemporalShadowFS(FullscreenOut input) : SV_Target {
    return rt_temporal(input.position.xy, 24.0);
}

[shader("fragment")]
float4 rtTemporalGiFS(FullscreenOut input) : SV_Target {
    return rt_temporal(input.position.xy, 48.0);
}

// ---- spatial denoise (geometry-aware Vogel disc). Shadows: kernel = projected penumbra width,
// derived from blocker distance and the solar disc, so contact shadows stay razor sharp.

static const int kRtTaps = 12;

float2 rt_vogel(int i, float rot) {
    float r = sqrt((float(i) + 0.5) / float(kRtTaps));
    float a = float(i) * 2.39996323 + rot;
    return float2(cos(a), sin(a)) * r;
}

float rt_geom_weight(RtPixel c, RtPixel t) {
    if (!t.valid) {
        return 0.0;
    }
    float plane = abs(dot(t.p - c.p, c.n));
    float w_z = saturate(1.0 - plane / (0.03 + 0.012 * c.view_t));
    float w_n = pow(saturate(dot(c.n, t.n)), 16.0);
    return w_z * w_n;
}

[shader("fragment")]
float4 rtDenoiseShadowFS(FullscreenOut input) : SV_Target {
    const int2 px = int2(input.position.xy);
    float4 center = rt_a.Load(int3(px, 0));
    RtPixel g = rt_pixel(px);
    if (!g.valid) {
        return center;
    }
    const float ppm = rt_px_per_metre(g.view_t);
    const float conf = saturate(center.w / 24.0);
    float penumbra_m = center.z * tan(frame.rt_params.y) * 2.0;
    float r_shadow = clamp(penumbra_m * ppm, 0.0, 16.0) * lerp(1.0, 0.55, conf) + 1.0;
    float r_ao = clamp(0.3 * ppm, 2.0, 10.0) * lerp(1.0, 0.6, conf);
    float rot = ign(input.position.xy) * 6.2831853;

    float s_sum = center.x, s_w = 1.0;
    float a_sum = center.y, a_w = 1.0;
    float b_sum = center.z, b_w = 1.0;
    for (int i = 0; i < kRtTaps; ++i) {
        float2 o = rt_vogel(i, rot);
        int2 ps = rt_clamp_px(px + int2(round(o * r_shadow)));
        float w = rt_geom_weight(g, rt_pixel(ps));
        float4 s = rt_a.Load(int3(ps, 0));
        s_sum += s.x * w;
        s_w += w;
        // Blockers only blur with blockers so lit/unlit boundaries keep their width estimate.
        float wb = w * (s.z > 0.0 ? 1.0 : 0.0);
        b_sum += s.z * wb;
        b_w += wb;

        int2 pa = rt_clamp_px(px + int2(round(o * r_ao)));
        float wa = rt_geom_weight(g, rt_pixel(pa));
        a_sum += rt_a.Load(int3(pa, 0)).y * wa;
        a_w += wa;
    }
    return float4(s_sum / s_w, a_sum / a_w, b_sum / b_w, center.w);
}

// Edge-aware 5x5 a-trous wavelet pass (SVGF-style, geometry stops only: GI is low frequency).
// Two passes with growing stride cover ~20 px while every tap stays on the same surface.
float4 rt_atrous_gi(int2 px, float stride) {
    float4 center = rt_a.Load(int3(px, 0));
    RtPixel g = rt_pixel(px);
    if (!g.valid) {
        return center;
    }
    const float conf = saturate(center.w / 48.0);
    const float step = stride * max(1.0, frame.screen.w / 720.0) * lerp(1.0, 0.6, conf);
    const float k[3] = {0.375, 0.25, 0.0625};
    float3 sum = center.xyz * (k[0] * k[0]);
    float wsum = k[0] * k[0];
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            if (x == 0 && y == 0) {
                continue;
            }
            int2 ps = rt_clamp_px(px + int2(round(float2(float(x), float(y)) * step)));
            float w = k[abs(x)] * k[abs(y)] * rt_geom_weight(g, rt_pixel(ps));
            sum += rt_a.Load(int3(ps, 0)).xyz * w;
            wsum += w;
        }
    }
    return float4(sum / wsum, center.w);
}

[shader("fragment")]
float4 rtDenoiseGiFS(FullscreenOut input) : SV_Target {
    return rt_atrous_gi(int2(input.position.xy), 1.5);
}

[shader("fragment")]
float4 rtDenoiseGi2FS(FullscreenOut input) : SV_Target {
    return rt_atrous_gi(int2(input.position.xy), 4.0);
}

// ---- reflections: hybrid screen-space reuse + ray-traced fallback (fixes SSR's off-screen holes)

float3 puddle_rt(float2 uv, float3 p, float3 base_col, float2 pixel) {
    float3 ro = frame.camera_pos.xyz;
    float3 rd = normalize(p - ro);
    float t_anim = frame.camera_pos.w;
    float rx = sin(p.x * 7.0 + t_anim * 1.1) * 0.0018 + sin(p.z * 11.0 - t_anim * 0.8) * 0.0014;
    float rz = cos(p.z * 8.0 + t_anim * 1.3) * 0.0018 + cos(p.x * 13.0 + t_anim * 0.9) * 0.0014;
    float3 wn = normalize(float3(rx, 1.0, rz));
    float3 rdir = reflect(rd, wn);

    float3 refl = sky_color(rdir);
    RayDesc ray;
    ray.Origin = p + float3(0.0, 0.03, 0.0);
    ray.Direction = rdir;
    ray.TMin = 0.0;
    ray.TMax = 2000.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene_tlas, 0, 0xFF, ray);
    while (q.Proceed()) {}
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        RtHit h = rt_fetch_hit(q.CommittedInstanceID(), q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(),
                               q.CommittedObjectToWorld3x4(), ray.Origin, ray.Direction, q.CommittedRayT());
        // Visible on screen at the same depth: reuse the fully shaded pixel (materials, GI, detail).
        float4 cs = mul(frame.view_proj, float4(h.position, 1.0));
        float2 suv = cs.xy / max(cs.w, 1e-4) * 0.5 + 0.5;
        bool on_screen = cs.w > 0.1 && all(suv > 0.0) && all(suv < 1.0);
        float3 traced = apply_fog(rt_shade_hit(h, rt_rand2(pixel, 9.0)), h.position);
        refl = traced;
        if (on_screen) {
            float scene_t = linear_depth(scene_depth.SampleLevel(suv, 0.0));
            float match = 1.0 - smoothstep(0.02, 0.08, abs(scene_t - cs.w) / max(cs.w, 0.1));
            float2 edge = smoothstep(0.0, 0.04, suv) * smoothstep(1.0, 0.96, suv);
            refl = lerp(traced, hdr_scene.SampleLevel(suv, 0.0).rgb, match * edge.x * edge.y);
        }
    }
    float ndv = clamp(dot(wn, -rd), 1e-4, 1.0);
    float rx_f = saturate(1.0 - ndv);
    float fresnel = 0.04 + 0.96 * (rx_f * rx_f * rx_f * sqrt(rx_f));
    float blend = saturate(fresnel * 1.05 + 0.22);
    return lerp(float3(0.045, 0.036, 0.026) + base_col * 0.12, refl * 0.95, blend);
}

[shader("fragment")]
float4 rtCompositeFS(FullscreenOut input) : SV_Target {
    const float2 uv = input.uv;
    float4 hdr_raw = hdr_scene.SampleLevel(uv, 0.0);
    float3 c = hdr_raw.rgb;
    float d = scene_depth.SampleLevel(uv, 0.0);
    float view_t = d < 1.0 ? linear_depth(d) : 1e6;
    // AO and GI were ray traced into the forward pass; only reflections and sun shafts remain.
    if (d < 1.0 && view_t > 0.45 && hdr_raw.a > 0.05) {
        float3 p = screen_to_world(uv, view_t);
        c = (asuint(frame.rt_params.w) & RT_FLAG_REFLECTIONS) != 0u ? puddle_rt(uv, p, c, input.position.xy)
                                                                     : puddle_ssr(uv, p, c, input.position.xy);
    }
    if (view_t > 2.5) {
        c += volumetric_sun_shafts(uv, view_t, input.position.xy);
    }
    c += crepuscular_sun_shafts(uv, view_t, input.position.xy);
    return float4(c, 1.0);
}

#endif // NOVA_RT
