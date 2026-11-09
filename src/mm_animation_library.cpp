#include "mm_animation_library.h"

#include "common.h"
#include "features/mm_feature.h"
#include "math/hash.h"
#include "math/stats.hpp"
#include "mm_character.h"

#include <algorithm>
#include <limits>

void MMAnimationLibrary::bake_data(const MMCharacter* p_character, const AnimationMixer* p_player, const Skeleton3D* p_skeleton) {
    ERR_FAIL_COND(features.is_empty());

    motion_data.clear();
    db_anim_index.clear();
    db_time_index.clear();
    db_pose_offset.clear();

    _build_tag_data();
    ERR_FAIL_COND_MSG(tag_names.size() > MAX_TAGS, vformat("Too many tags in the library (%d, max %d).", tag_names.size(), MAX_TAGS));

    int32_t dim_count = 0;
    for (auto i = 0; i < features.size(); ++i) {
        MMFeature* f = Object::cast_to<MMFeature>(features[i]);
        // Setup first: it can change the dimension count (e.g. trajectory point counts
        // are copied from the character).
        f->setup_skeleton(p_character, p_player, p_skeleton);
        dim_count += f->get_dimension_count();
    }

    TypedArray<StringName> animation_list = get_animation_list();

    // Normalization data
    std::vector<std::vector<StatsAccumulator>> stats(features.size());

    PackedFloat32Array data;
    int32_t current_pose_offset = 0;
    // For every animation
    for (int64_t animation_index = 0; animation_index < animation_list.size(); animation_index++) {
        const StringName& anim_name = animation_list[animation_index];
        Ref<Animation> animation = get_animation(anim_name);
        db_pose_offset.push_back(current_pose_offset);

        // Initialize features
        for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
            MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
            stats[feature_index].resize(feature->get_dimension_count());
            feature->setup_for_animation(animation);
        }

        const double animation_length = get_baked_length(animation);
        const double time_step = 1.0f / get_sampling_rate();
        int pose_count = 0;
        // Every time step
        for (double time = 0; time < animation_length; time += time_step) {
            PackedFloat32Array pose_data;
            // For every feature
            for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
                const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
                const PackedFloat32Array feature_data = feature->bake_animation_pose(animation, time);

                ERR_FAIL_COND(feature_data.size() != feature->get_dimension_count());

                // Update stats
                for (int64_t feature_element_index = 0; feature_element_index < feature_data.size(); feature_element_index++) {
                    stats[feature_index][feature_element_index].add_sample(feature_data[feature_element_index]);
                }

                pose_data.append_array(feature_data);
                current_pose_offset += feature->get_dimension_count();
            }

            ERR_FAIL_COND(pose_data.size() != dim_count);

            // Update dataset
            data.append_array(pose_data);
            db_anim_index.push_back(animation_index);
            db_time_index.push_back(time);
            pose_count++;
        }
    }

    // Compute mean and standard deviation
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);

        PackedFloat32Array feature_means;
        feature_means.resize(feature->get_dimension_count());
        PackedFloat32Array feature_std_devs;
        feature_std_devs.resize(feature->get_dimension_count());
        PackedFloat32Array feature_mins;
        feature_mins.resize(feature->get_dimension_count());
        PackedFloat32Array feature_maxes;
        feature_maxes.resize(feature->get_dimension_count());

        for (int64_t feature_element_index = 0; feature_element_index < feature->get_dimension_count(); feature_element_index++) {
            feature_means.set(feature_element_index, stats[feature_index][feature_element_index].get_mean());
            feature_std_devs.set(feature_element_index, stats[feature_index][feature_element_index].get_standard_deviation());
            feature_mins.set(feature_element_index, stats[feature_index][feature_element_index].get_min());
            feature_maxes.set(feature_element_index, stats[feature_index][feature_element_index].get_max());
        }

        feature->set_means(feature_means);
        feature->set_std_devs(feature_std_devs);
        feature->set_mins(feature_mins);
        feature->set_maxes(feature_maxes);
    }

    _normalize_data(data, dim_count);

    motion_data = data.duplicate();

    schema_hash = compute_features_hash();

    _build_kd_trees(dim_count);
}

PackedStringArray MMAnimationLibrary::get_animation_meta_tags(const Ref<Animation>& p_animation) {
    PackedStringArray tags;
    if (p_animation.is_null() || !p_animation->has_meta(TAGS_META)) {
        return tags;
    }
    const Variant meta = p_animation->get_meta(TAGS_META);
    if (meta.get_type() == Variant::PACKED_STRING_ARRAY) {
        tags = meta;
    } else if (meta.get_type() == Variant::ARRAY) {
        const Array array = meta;
        for (int64_t i = 0; i < array.size(); i++) {
            tags.push_back(String(array[i]));
        }
    } else if (meta.get_type() == Variant::STRING || meta.get_type() == Variant::STRING_NAME) {
        tags.push_back(String(meta));
    } else {
        ERR_PRINT(vformat("%s metadata must be a PackedStringArray.", TAGS_META));
    }
    return tags;
}

void MMAnimationLibrary::_build_tag_data() {
    const TypedArray<StringName> animation_list = get_animation_list();

    // Vocabulary: every tag in use, sorted so the bits don't depend on clip order.
    PackedStringArray names;
    for (int64_t i = 0; i < animation_list.size(); i++) {
        const PackedStringArray tags = get_animation_meta_tags(get_animation(animation_list[i]));
        for (const String& tag : tags) {
            if (!tag.is_empty() && !names.has(tag)) {
                names.push_back(tag);
            }
        }
    }
    names.sort();
    tag_names = names;
    if (tag_names.size() > MAX_TAGS) {
        return;
    }

    db_anim_tag_masks.clear();
    for (int64_t i = 0; i < animation_list.size(); i++) {
        int64_t mask = 0;
        _tags_to_mask(get_animation_meta_tags(get_animation(animation_list[i])), mask);
        db_anim_tag_masks.push_back(mask);
    }
}

void MMAnimationLibrary::_build_kd_trees(int32_t p_dim_count) {
    _kd_trees.clear();
    _warned_masks.clear();
    node_indices.clear();
    kd_group_masks.clear();
    kd_group_offsets.clear();

    // Group the poses by their animation's tag mask, in order of first appearance.
    std::vector<std::vector<int>> group_poses;
    for (int64_t pose = 0; pose < db_anim_index.size(); pose++) {
        const int32_t animation_index = db_anim_index[pose];
        const int64_t mask = animation_index < db_anim_tag_masks.size() ? db_anim_tag_masks[animation_index] : 0;
        int64_t group = kd_group_masks.find(mask);
        if (group < 0) {
            group = kd_group_masks.size();
            kd_group_masks.push_back(mask);
            group_poses.emplace_back();
        }
        group_poses[group].push_back(static_cast<int>(pose));
    }

    for (const std::vector<int>& poses : group_poses) {
        kd_group_offsets.push_back(node_indices.size());
        std::unique_ptr<KDTree> tree = std::make_unique<KDTree>(motion_data.ptr(), p_dim_count, poses);
        node_indices.append_array(tree->get_node_indices());
        _kd_trees.push_back(std::move(tree));
    }
    kd_group_offsets.push_back(node_indices.size());
}

void MMAnimationLibrary::_ensure_kd_trees(int32_t p_dim_count) {
    if (!_kd_trees.empty()) {
        return;
    }
    // Baked before tags existed: node_indices is one tree over every pose.
    if (kd_group_offsets.size() < 2) {
        kd_group_masks.clear();
        kd_group_masks.push_back(0);
        kd_group_offsets.clear();
        kd_group_offsets.push_back(0);
        kd_group_offsets.push_back(node_indices.size());
    }
    for (int64_t group = 0; group + 1 < kd_group_offsets.size(); group++) {
        std::unique_ptr<KDTree> tree = std::make_unique<KDTree>(p_dim_count);
        tree->rebuild_tree(0, node_indices.slice(kd_group_offsets[group], kd_group_offsets[group + 1]));
        _kd_trees.push_back(std::move(tree));
    }
}

bool MMAnimationLibrary::_tags_to_mask(const PackedStringArray& p_tags, int64_t& r_mask) const {
    r_mask = 0;
    bool all_known = true;
    for (const String& tag : p_tags) {
        if (tag.is_empty()) {
            continue;
        }
        const int64_t bit = tag_names.find(tag);
        if (bit < 0 || bit >= MAX_TAGS) {
            all_known = false;
            continue;
        }
        r_mask |= int64_t(1) << bit;
    }
    return all_known;
}

bool MMAnimationLibrary::_is_animation_eligible(int64_t p_animation_index, int64_t p_required_mask, int64_t p_excluded_mask) const {
    const int64_t mask = p_animation_index >= 0 && p_animation_index < db_anim_tag_masks.size() ? db_anim_tag_masks[p_animation_index] : 0;
    return (mask & p_required_mask) == p_required_mask && (mask & p_excluded_mask) == 0;
}

PackedStringArray MMAnimationLibrary::get_animation_tags(const StringName& p_animation) const {
    PackedStringArray tags;
    const int64_t animation_index = get_animation_list().find(p_animation);
    if (animation_index < 0 || animation_index >= db_anim_tag_masks.size()) {
        return tags;
    }
    const int64_t mask = db_anim_tag_masks[animation_index];
    for (int64_t bit = 0; bit < tag_names.size() && bit < MAX_TAGS; bit++) {
        if (mask & (int64_t(1) << bit)) {
            tags.push_back(tag_names[bit]);
        }
    }
    return tags;
}

void MMAnimationLibrary::bake(MMCharacter* p_character, AnimationMixer* p_mixer, Skeleton3D* p_skeleton) {
    ERR_FAIL_NULL_MSG(p_character, "bake: character is null");
    ERR_FAIL_NULL_MSG(p_mixer, "bake: mixer is null");
    ERR_FAIL_NULL_MSG(p_skeleton, "bake: skeleton is null");
    bake_data(p_character, p_mixer, p_skeleton);
}

MMQueryOutput MMAnimationLibrary::query(const MMQueryInput& p_query_input, const String& p_current_animation, double p_current_time) {
    PackedFloat32Array query_vector = PackedFloat32Array();
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        if (!feature) {
            continue;
        }
        PackedFloat32Array feature_data = feature->evaluate_runtime_data(p_query_input);
        feature->normalize(feature_data.ptrw());
        for (int64_t i = 0; i < feature_data.size(); i++) {
            if (!Math::is_finite(feature_data[i])) {
                ERR_PRINT_ONCE(vformat("Motion matching query skipped: %s produced a non-finite value.", feature->get_class()));
                return MMQueryOutput();
            }
        }
        query_vector.append_array(feature_data);
    }

    int64_t required_mask = 0;
    int64_t excluded_mask = 0;
    if (!_tags_to_mask(p_query_input.required_tags, required_mask)) {
        // A required tag no clip has: nothing can match.
        return MMQueryOutput();
    }
    _tags_to_mask(p_query_input.excluded_tags, excluded_mask); // unknown excluded tags exclude nothing

    MMQueryOutput result;
    result = _search_kd_tree(query_vector, required_mask, excluded_mask);
    if (result.matched_pose_index < 0) {
        return result;
    }
    if (!include_cost_results) {
        result.cost = _compute_feature_costs(result.matched_pose_index, query_vector, nullptr);
    }
    if (!p_current_animation.is_empty()) {
        const int64_t current_pose = get_pose_index(p_current_animation, p_current_time);
        // A clip the tags rule out is no option to continue (current_cost stays -1).
        if (current_pose >= 0 && _is_animation_eligible(db_anim_index[current_pose], required_mask, excluded_mask)) {
            result.current_cost = _compute_feature_costs(current_pose, query_vector, nullptr);
        }
    }
    return result;
}

int64_t MMAnimationLibrary::get_pose_index(const String& p_animation, double p_time) const {
    // Matched names are "<library>/<animation>".
    const String name = p_animation.get_slice("/", p_animation.get_slice_count("/") - 1);
    const int64_t animation_index = get_animation_list().find(StringName(name));
    const int64_t dim_count = get_dim_count();
    if (animation_index < 0 || dim_count == 0 || animation_index >= db_pose_offset.size()) {
        return -1;
    }
    const int64_t start = db_pose_offset[animation_index] / dim_count;
    const int64_t end = animation_index + 1 < db_pose_offset.size()
        ? db_pose_offset[animation_index + 1] / dim_count
        : db_anim_index.size();
    if (end <= start) {
        return -1;
    }
    const int64_t offset = static_cast<int64_t>(Math::floor(MAX(p_time, 0.0) * sampling_rate));
    return start + CLAMP(offset, int64_t(0), end - start - 1);
}

int64_t MMAnimationLibrary::get_dim_count() const {
    int64_t dim_count = 0;
    for (auto i = 0; i < features.size(); ++i) {
        MMFeature* f = Object::cast_to<MMFeature>(features[i]);
        dim_count += f->get_dimension_count();
    }

    return dim_count;
}

int64_t MMAnimationLibrary::get_animation_pose_count(String p_animation_name) const {
    TypedArray<StringName> animation_list = get_animation_list();
    Ref<Animation> animation = get_animation(p_animation_name);
    if (animation.is_null()) {
        return 0;
    };
    const double animation_length = get_baked_length(animation);
    const double time_step = 1.0f / get_sampling_rate();
    return static_cast<int32_t>(UtilityFunctions::ceil(animation_length / time_step - 1e-6));
}

double MMAnimationLibrary::get_baked_length(const Ref<Animation>& p_animation) const {
    const double length = p_animation->get_length();
    if (p_animation->get_loop_mode() != Animation::LOOP_NONE || end_margin <= 0.f) {
        return length;
    }
    return MAX(length - end_margin, 1.0 / get_sampling_rate());
}

void MMAnimationLibrary::display_data(const Ref<EditorNode3DGizmo>& p_gizmo, const Transform3D& p_transform, String p_animation_name, int32_t p_pose_index) const {
    const int32_t anim_index = get_animation_list().find(p_animation_name);
    const int32_t dim_count = get_dim_count();
    int32_t start_frame_index = db_pose_offset[anim_index];
    int32_t frame_index = start_frame_index + p_pose_index * dim_count;

    for (size_t feature_index = 0; feature_index < features.size(); feature_index++) {
        const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        const float* frame_motion_data = motion_data.ptr() + frame_index;
        feature->display_data(p_gizmo, p_transform, frame_motion_data);
        frame_index += feature->get_dimension_count();
    }
}

int64_t MMAnimationLibrary::compute_features_hash() const {
    int64_t hash = 0;
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        TypedArray<Dictionary> feature_properties = feature->get_property_list();
        for (int64_t property_index = 0; property_index < feature_properties.size(); property_index++) {
            // TODO: This needs work, but works for now
            const Dictionary feature_property = Dictionary(feature_properties[property_index]);
            const String property_name = feature_property["name"];
            const uint32_t property_type = feature_property["type"];
            const bool property_is_stats =
                property_name == "means" ||
                property_name == "std_devs" ||
                property_name == "mins" ||
                property_name == "maxes";
            const bool is_resource_property = property_name.contains("resource");
            if (property_type != Variant::OBJECT &&
                property_type != Variant::NIL &&
                !property_is_stats &&
                !is_resource_property) {
                hash = hash_combine(hash, property_name.hash());
                Variant value = feature->get(property_name);
                // Hash floats at the precision they are stored with: a .tres keeps 0.33, which
                // reloads as a different double than the baked float (0.33000001...), and the
                // library would always look out of date.
                if (value.get_type() == Variant::FLOAT) {
                    value = static_cast<double>(static_cast<float>(static_cast<double>(value)));
                }
                hash = hash_combine(hash, value.hash());
            }
        }
    }
    return hash;
}

bool MMAnimationLibrary::needs_baking() const {
    if (schema_hash != compute_features_hash()) {
        return true;
    }
    // Tags changed since the bake?
    const TypedArray<StringName> animation_list = get_animation_list();
    for (int64_t i = 0; i < animation_list.size(); i++) {
        const PackedStringArray tags = get_animation_meta_tags(get_animation(animation_list[i]));
        const int64_t baked = i < db_anim_tag_masks.size() ? db_anim_tag_masks[i] : 0;
        int64_t mask = 0;
        if (!_tags_to_mask(tags, mask) || mask != baked) {
            return true;
        }
    }
    return false;
}

void MMAnimationLibrary::_normalize_data(PackedFloat32Array& p_data, size_t p_dim_count) const {
    ERR_FAIL_COND(p_data.is_empty() || (p_data.size() % p_dim_count != 0));

    for (int64_t frame_index = 0; frame_index < p_data.size(); frame_index += p_dim_count) {

        int dim_index = 0;
        for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
            const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
            feature->normalize(p_data.ptrw() + frame_index + dim_index);
            dim_index += feature->get_dimension_count();
        }
    }
}

float MMAnimationLibrary::_compute_feature_costs(int p_pose_index, const PackedFloat32Array& p_query, Dictionary* p_feature_costs) const {
    float pose_cost = 0.f;
    int start_frame_index = p_pose_index * p_query.size();
    int start_feature_index = 0;
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        if (!feature) {
            continue;
        }

        float feature_cost = 0.f;
        for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
            feature_cost += distance_squared((motion_data.ptr() + start_frame_index + start_feature_index + feature_dim_index),
                                             (p_query.ptr() + start_feature_index + feature_dim_index),
                                             1) *
                feature->calculate_normalized_weight(feature_dim_index);
        }

        if (p_feature_costs) {
            p_feature_costs->get_or_add(feature->get_class(), feature_cost);
        }

        pose_cost += feature_cost;
        start_feature_index += feature->get_dimension_count();
    }
    return pose_cost;
}

MMQueryOutput MMAnimationLibrary::_search_naive(const PackedFloat32Array& p_query) const {
    float cost = FLT_MAX;
    MMQueryOutput result;
    TypedArray<StringName> animation_list = get_animation_list();
    int32_t dim_count = p_query.size();
    int64_t best_pose_index = -1;
    for (int64_t start_frame_index = 0; start_frame_index < motion_data.size(); start_frame_index += dim_count) {
        int start_feature_index = start_frame_index;
        float pose_cost = 0.f;
        Dictionary feature_costs;
        for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
            const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
            if (!feature) {
                continue;
            }

            float feature_cost = 0.f;
            for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
                feature_cost += distance_squared((motion_data.ptr() + start_frame_index + start_feature_index + feature_dim_index),
                                                 (p_query.ptr() + start_feature_index + feature_dim_index),
                                                 1) *
                    feature->calculate_normalized_weight(feature_dim_index);
            }

            feature_costs.get_or_add(feature->get_class(), feature_cost);
            pose_cost += feature_cost;
            start_feature_index += feature->get_dimension_count();
        }

        if (pose_cost < cost) {
            cost = pose_cost;
            best_pose_index = start_frame_index / dim_count;
        }
    }

    String library_name = get_path().get_file().get_basename() + "/";
    if (library_name.is_empty()) {
        library_name = get_name() + "/";
    }

    result.matched_pose_index = best_pose_index;
    result.animation_match = library_name + UtilityFunctions::str(animation_list[db_anim_index[best_pose_index]]);
    result.time_match = db_time_index[best_pose_index];

    if (include_cost_results) {
        result.matched_frame_data = motion_data.slice(
            best_pose_index * dim_count,
            (best_pose_index + 1) * dim_count);

        result.cost = _compute_feature_costs(
            best_pose_index,
            p_query,
            &result.feature_costs);
    }

    return result;
}

MMQueryOutput MMAnimationLibrary::_search_kd_tree(const PackedFloat32Array& p_query, int64_t p_required_mask, int64_t p_excluded_mask) {
    const int32_t dim_count = p_query.size();
    _ensure_kd_trees(dim_count);

    std::vector<float> dimension_weights;
    for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
        const MMFeature* feature = Object::cast_to<MMFeature>(features[feature_index]);
        if (!feature) {
            continue;
        }
        for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
            dimension_weights.push_back(feature->calculate_normalized_weight(feature_dim_index));
        }
    }

    // Search the tree of every group the tags allow; keep the cheapest pose.
    int best_pose_index = -1;
    float best_cost = std::numeric_limits<float>::max();
    for (size_t group = 0; group < _kd_trees.size(); group++) {
        const int64_t mask = kd_group_masks[group];
        if ((mask & p_required_mask) != p_required_mask || (mask & p_excluded_mask) != 0) {
            continue;
        }
        const int pose = _kd_trees[group]->search_nn(motion_data.ptr(), p_query.ptr(), dimension_weights);
        if (pose < 0 || pose >= db_anim_index.size()) {
            continue;
        }
        float cost = 0.f;
        const float* pose_data = motion_data.ptr() + int64_t(pose) * dim_count;
        for (int32_t d = 0; d < dim_count; d++) {
            const float diff = pose_data[d] - p_query[d];
            cost += diff * diff * dimension_weights[d];
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_pose_index = pose;
        }
    }
    if (best_pose_index < 0) {
        const std::pair<int64_t, int64_t> masks(p_required_mask, p_excluded_mask);
        if (std::find(_warned_masks.begin(), _warned_masks.end(), masks) == _warned_masks.end()) {
            _warned_masks.push_back(masks);
            WARN_PRINT(vformat("Motion matching: no clip in %s matches the requested tags.", get_path()));
        }
        return MMQueryOutput();
    }

    MMQueryOutput result;
    String library_name = get_path().get_file().get_basename() + "/";
    if (library_name.is_empty()) {
        library_name = get_name() + "/";
    }
    TypedArray<StringName> animation_list = get_animation_list();

    result.matched_pose_index = best_pose_index;
    result.animation_match = library_name + UtilityFunctions::str(animation_list[db_anim_index[best_pose_index]]);
    result.time_match = db_time_index[best_pose_index];

    if (include_cost_results) {
        result.matched_frame_data = motion_data.slice(
            best_pose_index * p_query.size(),
            (best_pose_index + 1) * p_query.size());

        result.cost =
            _compute_feature_costs(
                best_pose_index,
                p_query,
                &result.feature_costs);
    }

    return result;
}

void MMAnimationLibrary::_bind_methods() {
    ClassDB::bind_method(D_METHOD("bake", "character", "mixer", "skeleton"), &MMAnimationLibrary::bake);
    ClassDB::bind_method(D_METHOD("needs_baking"), &MMAnimationLibrary::needs_baking);
    ClassDB::bind_method(D_METHOD("get_animation_tags", "animation"), &MMAnimationLibrary::get_animation_tags);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::ARRAY, features, PROPERTY_HINT_TYPE_STRING, UtilityFunctions::str(Variant::OBJECT) + '/' + UtilityFunctions::str(Variant::BASIS) + ":MMFeature");
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::FLOAT, sampling_rate);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::BOOL, include_cost_results);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::FLOAT, end_margin);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_FLOAT32_ARRAY, motion_data, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT32_ARRAY, db_anim_index, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_FLOAT32_ARRAY, db_time_index, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_FLOAT32_ARRAY, db_pose_offset, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::INT, schema_hash, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT32_ARRAY, node_indices, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_STRING_ARRAY, tag_names, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT64_ARRAY, db_anim_tag_masks, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT64_ARRAY, kd_group_masks, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
    BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT32_ARRAY, kd_group_offsets, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
}
