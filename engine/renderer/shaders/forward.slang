// Forward shading for opaque meshes: one directional light + ambient.
// Push constant layout must match renderer.cpp (128 bytes, the guaranteed minimum).

struct PushConstants {
    float4x4 mvp;
    // Columns of the model matrix's upper 3x3 (uniform scale assumed); w carries the
    // light direction (towards the light) in world space.
    float4 normal_col0;
    float4 normal_col1;
    float4 normal_col2;
    float4 color; // rgb = albedo (linear), a = ambient factor
};

[[vk::push_constant]] PushConstants pc;

struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
};

[shader("vertex")]
VSOutput vsMain(VSInput input) {
    VSOutput o;
    o.position = mul(pc.mvp, float4(input.position, 1.0));
    o.normal = pc.normal_col0.xyz * input.normal.x + pc.normal_col1.xyz * input.normal.y +
               pc.normal_col2.xyz * input.normal.z;
    return o;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target {
    const float3 n = normalize(input.normal);
    const float3 l = normalize(float3(pc.normal_col0.w, pc.normal_col1.w, pc.normal_col2.w));
    const float ambient = pc.color.a;
    const float diffuse = max(dot(n, l), 0.0);
    return float4(pc.color.rgb * (ambient + (1.0 - ambient) * diffuse), 1.0);
}
