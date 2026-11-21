# Third-party software

| Name | Version | License | Used by | Notes |
|------|---------|---------|---------|-------|
| [SDL3](https://github.com/libsdl-org/SDL) | 3.2.8 | zlib | `Nova::platform` (private) | Window/input backend, static, optional (`NOVA_WITH_SDL3`) |
| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | 1.4.335 | Apache-2.0 / MIT | `Nova::rhi` (private) | Headers only |
| [volk](https://github.com/zeux/volk) | 1.4.335 | MIT | `Nova::rhi` (private) | Vulkan meta-loader; no link-time loader dependency |
| [Slang](https://github.com/shader-slang/slang) | 2026.22.1 | Apache-2.0 with LLVM exception | build tool only | `slangc` compiles shaders to SPIR-V; not shipped |
| [doctest](https://github.com/doctest/doctest) | 2.4.11 | MIT | `tests/` only | Header-only unit test framework |
| [cgltf](https://github.com/jkuhlmann/cgltf) | 1.14 | MIT | `Nova::renderer` (private) | Header-only glTF 2.0 parser (glTF importer) |
| [stb](https://github.com/nothings/stb) | commit 6b5799b | MIT / public domain | `Nova::renderer` (private), `tests/` | `stb_image` PNG/JPEG decode; `stb_image_write` in tests only |

Pinned archives and SHA-256 hashes are in `cmake/NovaDependencies.cmake`. A `slangc` found on `PATH` or in
`$VULKAN_SDK/bin` is used instead of the download. Policy: `docs/adr/0006-third-party-dependencies.md`.

## Art assets

| Asset | Author | License | Used by |
|-------|--------|---------|---------|
| [FPS AK-74m animations](https://sketchfab.com/3d-models/fps-ak-74m-animations-94be8385c402474cacd39bc096c6ca14) (`assets/viewmodels/fps_ak74m.glb`) | [Cransh](https://sketchfab.com/ccransh) | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | `nova_guerrilla` first-person arms + rifle (unmodified) |
