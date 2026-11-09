#include "features/mm_curve_feature.h"

void MMCurveFeature::setup_for_animation(Ref<Animation> p_animation) {
    _value_track = -1;
    _bezier_track = -1;
    if (p_animation.is_null() || track_path.is_empty()) {
        return;
    }
    _value_track = p_animation->find_track(track_path, Animation::TYPE_VALUE);
    if (_value_track == -1) {
        _bezier_track = p_animation->find_track(track_path, Animation::TYPE_BEZIER);
    }
}

int64_t MMCurveFeature::get_dimension_count() const {
    return 1;
}

PackedFloat32Array MMCurveFeature::bake_animation_pose(Ref<Animation> p_animation, double p_time) const {
    float value = default_value;
    if (_value_track != -1) {
        value = static_cast<float>(p_animation->value_track_interpolate(_value_track, p_time));
    } else if (_bezier_track != -1) {
        value = p_animation->bezier_track_interpolate(_bezier_track, p_time);
    } else if (!curve_name.is_empty() && p_animation->has_meta(curve_name)) {
        value = static_cast<float>(p_animation->get_meta(curve_name));
    }

    PackedFloat32Array result;
    result.append(value);
    return result;
}

PackedFloat32Array MMCurveFeature::evaluate_runtime_data(const MMQueryInput& p_query_input) const {
    PackedFloat32Array result;
    result.append(static_cast<float>(p_query_input.feature_values.get(curve_name, default_value)));
    return result;
}

void MMCurveFeature::_bind_methods() {
    BINDER_PROPERTY_PARAMS(MMCurveFeature, Variant::STRING_NAME, curve_name);
    BINDER_PROPERTY_PARAMS(MMCurveFeature, Variant::NODE_PATH, track_path);
    BINDER_PROPERTY_PARAMS(MMCurveFeature, Variant::FLOAT, default_value);
}
