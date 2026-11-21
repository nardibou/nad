#pragma once

// Procedural first-person viewmodel of an AKM assault rifle held by two hands, with procedurally
// generated PBR textures (glTF metallic-roughness conventions).

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"
#include "nova/renderer/model.hpp"

#include <vector>

namespace guerrilla {

enum class ViewMaterial : nova::u8 { Gunmetal, Wood, Skin, Sleeve, Flash, Count };

struct ViewPiece {
    nova::renderer::MeshData mesh;
    ViewMaterial material;
};

// Each part is drawn with its own transform (so the magazine / left hand can animate on reload).
struct ViewmodelAssets {
    std::vector<ViewPiece> weapon;     // receiver, barrel, furniture, sights ... (no magazine)
    std::vector<ViewPiece> magazine;   // in weapon space, seated position
    std::vector<ViewPiece> right_hand; // sleeve + hand wrapped around the pistol grip, finger on the trigger
    std::vector<ViewPiece> left_hand;  // sleeve + hand under/around the lower handguard
    std::vector<ViewPiece> flash;      // muzzle flash (ViewMaterial::Flash), centred on `muzzle`, along -Z
    nova::math::Vec3 muzzle;           // tip of the muzzle brake
    nova::math::Vec3 rear_sight;       // top of the rear-sight notch
    nova::math::Vec3 front_sight;      // top of the front-sight post
    nova::math::Vec3 magazine_pivot;   // where the mag hooks into the receiver (front top of the mag)
    nova::math::Vec3 grip;             // centre of the pistol grip (right-hand anchor)
    // Procedural textures, indexed by material. sRGB: base_color. Linear: metallic_roughness
    // (G = roughness, B = metallic, R = 1) and normal (tangent space, green = +v / image down,
    // matching the engine's generated tangents: bitangent = cross(normal, tangent) * w = +v).
    struct Textures {
        nova::renderer::ImageData base_color, metallic_roughness, normal;
    };
    Textures textures[static_cast<int>(ViewMaterial::Count)];
    // Scalar factors to use with the textures, per material. Metallic/roughness are 1 because the
    // textures already hold the final values.
    struct Factors {
        float metallic, roughness;
        nova::math::Vec3 emissive;
        bool double_sided;
    };
    Factors factors[static_cast<int>(ViewMaterial::Count)];
};

// Deterministic. Weapon space: metres, +X right, +Y up, -Z forward (muzzle direction).
// Origin = centre of the rear of the receiver at the height of the bore axis (bore axis is the
// line y = 0, x = 0). The line of sight (rear_sight -> front_sight) is parallel to -Z above the bore.
[[nodiscard]] ViewmodelAssets build_ak_viewmodel();

} // namespace guerrilla
