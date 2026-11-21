// NOVA Craft 2D Crisp UI & 3D Gizmo Overlay Shader (native 2K/4K sRGB-accurate alpha blending).

struct UiPush {
    float2 inv_size;
    float2 _pad;
};
[[vk::push_constant]] ConstantBuffer<UiPush> ui_push;

[[vk::binding(0, 0)]] Sampler2D<float4> ui_tex;

struct VSInput {
    [[vk::location(0)]] float2 pos;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] float4 color;
};

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

[shader("vertex")]
VSOutput uiVS(VSInput input) {
    VSOutput o;
    float2 ndc = input.pos * ui_push.inv_size * 2.0 - 1.0;
    o.position = float4(ndc, 0.0, 1.0);
    o.uv = input.uv;
    o.color = input.color;
    return o;
}

float3 srgb_to_linear(float3 c) {
    return pow(max(c, 0.0), 2.2);
}

[shader("fragment")]
float4 uiFS(VSOutput input) : SV_Target {
    float4 tex = ui_tex.Sample(input.uv);
    float3 col = srgb_to_linear(input.color.rgb) * tex.rgb;
    float a = input.color.a * tex.a;
    return float4(col, a);
}
