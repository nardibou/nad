# 9. NOVA Lumina RT: hybrid ray-traced lighting

- Status: Accepted
- Date: 2026-11-21

## Context
The raster lighting stack (cached sun shadow map + PCSS, height-map contact AO, screen-space GTAO/SSGI,
screen-space puddle reflections) hits the limits every screen-space or shadow-map technique has:

- shadow maps alias, need per-scene bias tuning and cover a fixed radius (distant casters are lost);
- screen-space AO/GI/reflections only see what is on screen, so occluders and bounce sources that leave
  the view vanish, and haloes form around silhouettes;
- the hemisphere "ambient" ignores the actual scene, so shaded areas read flat and cold.

Unreal Engine 5 addresses this with Lumen and Virtual Shadow Maps, which still lean on approximations
(mesh/global distance fields, a surface cache, screen traces, page-cached shadow maps). Those cause well
known artefacts: light leaking through thin walls, missing small or thin occluders, ghosting from long
temporal accumulation, and shadow-page invalidation cost with animated geometry. Hardware ray tracing is now
widely available (inline ray queries in Vulkan 1.2+), including in Mesa's lavapipe, so CI can test it.

## Decision
- **RHI**: bottom-level acceleration structures (`create_blas`, one per static mesh, built at upload),
  top-level structures (`create_tlas`, `CommandList::build_tlas`), `BufferUsage::Storage`,
  `BindingType::{StorageBuffer, AccelerationStructure}` and `buffer_address()` for shader pointers.
  Still no Vulkan types in public headers. Support is detected per device (`VK_KHR_acceleration_structure`,
  `VK_KHR_ray_query`, `bufferDeviceAddress`) and reported as `DeviceInfo::ray_tracing`. `NOVA_NO_RT=1` or
  `DeviceDesc::ray_tracing = false` turns it off.
- **Shaders**: `world.slang` is compiled twice. `world_rt_spv` (`-DNOVA_RT=1`, SPIR-V 1.4) adds the ray
  query passes and is only loaded on ray-tracing devices. Without support the raster path is unchanged.
- **Inline ray queries in fragment passes**, not ray-tracing pipelines: no shader binding table, they fit
  the existing fullscreen-pass structure, and they run on every RT-capable vendor and on lavapipe.
- **Frame structure (RT devices)**:
  1. TLAS rebuild only when the instance set changes (hashed per frame-in-flight slot).
  2. Normal + depth prepass (normals face the viewer: robust to two-sided meshes with reversed winding).
  3. Sun visibility: one ray per pixel towards a point on the **solar disc** (`sun_angular_radius`). This
     gives physically correct, contact-hardening penumbrae with no bias tuning and no fixed coverage radius
     (`rt_shadow_distance`). A short cosine-distributed AO ray runs in the same pass.
  4. Diffuse GI: cosine rays, hit shading from the vertex data (read through buffer-device-address
     pointers) with its own sun shadow ray, and sky radiance on a miss. This is real one-bounce GI on real
     triangles, not a distance-field or surface-cache approximation, so thin walls do not leak.
  5. Temporal accumulation per signal: depth-validated reprojection with the previous view-projection plus a
     3x3 neighbourhood clamp (short histories: 24 frames for shadows/AO, 48 for GI), and a history reset on
     lighting discontinuities (sun jumps > ~1.8°, preset switches). This keeps ghosting low.
  6. Spatial denoising: shadows use a Vogel-disc kernel sized by the **projected penumbra width**
     (blocker distance × solar disc), so contact shadows stay sharp and distant ones soft. GI uses two
     edge-aware à-trous passes.
  7. The forward pass reads the denoised signals per pixel. Water reflections trace the TLAS and reuse the
     on-screen shaded pixel when the hit is visible (screen-space quality, no SSR holes).
- **Hybrid casters**: skinned/dynamic meshes and alpha-tested assets are not in the TLAS yet. They render
  into a small per-frame shadow map that is multiplied in, so nothing stops casting shadows.
- The cached full shadow map remains for volumetric sun shafts.

## Consequences
- Lighting quality no longer depends on screen content or shadow-map resolution. Debug views
  (`WorldSettings::rt_debug_view`, Guerrilla `--rt-view 1|2|3`) expose the denoised shadow, AO and GI.
- Cost: one prepass, three rays per pixel (shadow, AO, GI) plus one shadow ray per GI hit, and six
  fullscreen filter passes. This is intended for RT-class GPUs. Lower-end hardware keeps the raster path.
- 1 sample per pixel per signal relies on temporal accumulation; the first frames after a cut or a
  lighting jump are noisier until history builds up (about 0.25 s at 60 fps).

## Follow-ups
- BLAS refit for skinned meshes, alpha-tested any-hit with bindless textures, BLAS compaction.
- Half-resolution GI with upsampling, and ReSTIR-style reuse for many lights / emissive surfaces.
- Glossy reflections beyond water (roughness-aware cone tracing with the same denoiser).
