#pragma once

#include "algo/kd_tree.h"
#include "common.h"
#include "features/mm_feature.h"
#include "mm_query.h"

#include <godot_cpp/classes/animation_library.hpp>
#include <godot_cpp/classes/editor_node3d_gizmo.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/skeleton3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <memory>

class MMFeature;
class MMCharacter;

class MMAnimationLibrary : public AnimationLibrary {
    GDCLASS(MMAnimationLibrary, AnimationLibrary)

public:
    void bake_data(const MMCharacter* p_character, const AnimationMixer* p_player, const Skeleton3D* p_skeleton);
    // Script-facing wrapper around bake_data, so databases can be (re)built from tool scripts / CI.
    void bake(MMCharacter* p_character, AnimationMixer* p_mixer, Skeleton3D* p_skeleton);
    // p_current_animation / p_current_time: what is playing now, so the cost of continuing
    // it can be compared with the best match (MMQueryOutput::current_cost).
    MMQueryOutput query(const MMQueryInput& p_query_input, const String& p_current_animation = String(), double p_current_time = 0.0);
    int64_t get_pose_index(const String& p_animation, double p_time) const;
    int64_t get_dim_count() const;
    // Length of the part of p_animation that goes into the database (see end_margin).
    double get_baked_length(const Ref<Animation>& p_animation) const;
    int64_t get_animation_pose_count(String p_animation_name) const;

    void display_data(const Ref<EditorNode3DGizmo>& p_gizmo, const Transform3D& p_transform, String p_animation_name, int32_t p_pose_index) const;

    int64_t compute_features_hash() const;

    // Tags. A clip's tags come from its "mm_tags" metadata (PackedStringArray); queries only
    // return poses whose clip has every required tag and no excluded one.
    static constexpr const char* TAGS_META = "mm_tags";
    static constexpr int MAX_TAGS = 64;
    PackedStringArray get_animation_tags(const StringName& p_animation) const;
    // Tags of p_animation as stored in its metadata now (not as baked).
    static PackedStringArray get_animation_meta_tags(const Ref<Animation>& p_animation);

    bool needs_baking() const;

    GETSET(TypedArray<MMFeature>, features)
    GETSET(float, sampling_rate, 1.f)
    GETSET(bool, include_cost_results, false);
    // Seconds at the end of non-looping clips that are not added to the database. Those poses
    // have (almost) no future, and matching them leaves the character holding a clip's last
    // frame while it keeps moving.
    GETSET(float, end_margin, 0.f);

    // Database data
    GETSET(PackedFloat32Array, motion_data)
    GETSET(PackedInt32Array, db_anim_index)
    GETSET(PackedFloat32Array, db_time_index)
    GETSET(PackedInt32Array, db_pose_offset)
    GETSET(int64_t, schema_hash)

    // Tag data: every tag in use (bit i = tag_names[i]), and each animation's tag mask.
    GETSET(PackedStringArray, tag_names)
    GETSET(PackedInt64Array, db_anim_tag_masks)

    // KD Tree data: one tree per distinct tag mask ("group"), flattened one after another into
    // node_indices; group g is node_indices[kd_group_offsets[g] .. kd_group_offsets[g + 1]).
    GETSET(PackedInt32Array, node_indices);
    GETSET(PackedInt64Array, kd_group_masks)
    GETSET(PackedInt32Array, kd_group_offsets)

protected:
    static void _bind_methods();

private:
    void _normalize_data(PackedFloat32Array& p_data, size_t p_dim_count) const;
    float _compute_feature_costs(int p_pose_index, const PackedFloat32Array& p_query, Dictionary* p_feature_costs) const;

    MMQueryOutput _search_naive(const PackedFloat32Array& p_query) const;
    MMQueryOutput _search_kd_tree(const PackedFloat32Array& p_query, int64_t p_required_mask, int64_t p_excluded_mask);

    // Tag names -> mask. Returns false if a name isn't a tag of this library.
    bool _tags_to_mask(const PackedStringArray& p_tags, int64_t& r_mask) const;
    // Whether the animation's baked tags satisfy the masks.
    bool _is_animation_eligible(int64_t p_animation_index, int64_t p_required_mask, int64_t p_excluded_mask) const;
    void _build_tag_data();
    void _build_kd_trees(int32_t p_dim_count);
    void _ensure_kd_trees(int32_t p_dim_count);

    // One per kd group (same order as kd_group_masks).
    std::vector<std::unique_ptr<KDTree>> _kd_trees;
    // Tag sets already warned about (no eligible group), so the warning is printed once.
    std::vector<std::pair<int64_t, int64_t>> _warned_masks;
};
