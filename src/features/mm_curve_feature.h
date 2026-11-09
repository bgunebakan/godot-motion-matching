#ifndef MM_CURVE_FEATURE_H
#define MM_CURVE_FEATURE_H

#include "common.h"
#include "features/mm_feature.h"

using namespace godot;

// Matches one named scalar (speed, stance, foot phase, ...) that the game sets at runtime.
class MMCurveFeature : public MMFeature {
    GDCLASS(MMCurveFeature, MMFeature)

public:
    virtual void setup_for_animation(Ref<Animation> p_animation) override;
    virtual int64_t get_dimension_count() const override;
    virtual PackedFloat32Array bake_animation_pose(Ref<Animation> p_animation, double p_time) const override;
    virtual PackedFloat32Array evaluate_runtime_data(const MMQueryInput& p_query_input) const override;

    // Key in MMCharacter.feature_values and in animation metadata.
    GETSET(StringName, curve_name);
    // Value or Bezier track sampled per frame when baking, e.g. "MotionCurves:speed".
    GETSET(NodePath, track_path);
    // Value used when neither the track nor the metadata exists.
    GETSET(float, default_value, 0.0f);

protected:
    static void _bind_methods();

private:
    int32_t _value_track = -1;
    int32_t _bezier_track = -1;
};

#endif // MM_CURVE_FEATURE_H
