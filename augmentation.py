"""
Data augmentation pipelines for WikiArt painting classification.

Paintings are NOT photographs — composition, brushstroke orientation, and
colour palette are part of an artist's style. Aggressive geometric distortions
(large rotations, shear, heavy zoom) can destroy exactly the features that
distinguish one artist from another. All pipelines below are designed with
this constraint in mind.

Usage:
    from src.augmentation import augmentation_conservative
    # ... then use as a layer inside a Model subclass
"""

from keras.layers import (
    GaussianNoise,
    RandomFlip,
    RandomBrightness,
    RandomContrast,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    Pipeline,
)


# --------------------------------------------------------------------------
# 1. Conservative — safest option, minimal risk of distorting style
# --------------------------------------------------------------------------
# Horizontal flip preserves artistic style (a mirrored painting still looks
# like the same artist). Tiny brightness variation simulates differences in
# digitisation/scanning conditions. No geometric distortion at all.
augmentation_conservative = Pipeline(
    [
        RandomFlip("horizontal"),
        RandomBrightness(factor=0.05, value_range=(0.0, 1.0)),
    ],
    name="augmentation_conservative",
)


# --------------------------------------------------------------------------
# 2. Mild — adds slight rotation to simulate imperfect scans
# --------------------------------------------------------------------------
# Same as conservative, plus a very small rotation (max ~5 degrees).
# Paintings in datasets are sometimes scanned slightly crooked, so this
# simulates realistic variation without altering the composition.
augmentation_mild = Pipeline(
    [
        RandomFlip("horizontal"),
        RandomBrightness(factor=0.05, value_range=(0.0, 1.0)),
        RandomRotation(factor=0.015, fill_mode="reflect"),
    ],
    name="augmentation_mild",
)


# --------------------------------------------------------------------------
# 3. Moderate — adds contrast variation and light zoom
# --------------------------------------------------------------------------
# Contrast variation simulates different monitor/scanner calibrations.
# Small zoom (±5%) simulates slightly different crops of the same painting,
# which is common across different digital reproductions of the same work.
augmentation_moderate = Pipeline(
    [
        RandomFlip("horizontal"),
        RandomBrightness(factor=0.08, value_range=(0.0, 1.0)),
        RandomContrast(factor=0.08),
        RandomRotation(factor=0.015, fill_mode="reflect"),
        RandomZoom((-0.05, 0.05), fill_mode="reflect"),
    ],
    name="augmentation_moderate",
)


# --------------------------------------------------------------------------
# 4. Moderate+ — adds small translation to reduce positional bias
# --------------------------------------------------------------------------
# Translation (±5%) shifts the painting slightly within the frame. This
# helps the model not rely on objects always being centred, which matters
# because digital reproductions may include varying amounts of border/frame.
augmentation_moderate_plus = Pipeline(
    [
        RandomFlip("horizontal"),
        RandomBrightness(factor=0.08, value_range=(0.0, 1.0)),
        RandomContrast(factor=0.08),
        RandomRotation(factor=0.02, fill_mode="reflect"),
        RandomZoom((-0.05, 0.05), fill_mode="reflect"),
        RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect"
        ),
    ],
    name="augmentation_moderate_plus",
)


# --------------------------------------------------------------------------
# 5. Aggressive — strongest option, still painting-aware
# --------------------------------------------------------------------------
# Pushes all parameters further. Use only if the model is still overfitting
# heavily after trying the milder options. The values are still well below
# what you would use for natural photographs (e.g. rotation is max ~10
# degrees, not 30+). Monitor val_loss closely — if it gets worse compared
# to moderate, this is too strong for the dataset.
augmentation_aggressive = Pipeline(
    [
        RandomFlip("horizontal"),
        RandomBrightness(factor=0.12, value_range=(0.0, 1.0)),
        RandomContrast(factor=0.12),
        RandomRotation(factor=0.03, fill_mode="reflect"),
        RandomZoom((-0.10, 0.10), fill_mode="reflect"),
        RandomTranslation(
            height_factor=0.08, width_factor=0.08, fill_mode="reflect"
        ),
    ],
    name="augmentation_aggressive",
)


# --------------------------------------------------------------------------
# 6. Moderate+ with noise — handles digitisation artifacts
# --------------------------------------------------------------------------
# Same geometric/colour transforms as moderate+, with added Gaussian noise.
# Paintings in this dataset come from diverse online sources — different
# scanners, cameras, JPEG compression levels, and web resolutions. A small
# amount of noise simulates these artifacts and helps the model not overfit
# to the specific digitisation quality of each image.
#
# This is especially relevant for artists with fine detail that is sensitive
# to compression: Gustave Dore's engravings, Albrecht Durer's line drawings,
# and Rembrandt's dark tonal work (all three have near-grayscale RGB profiles
# in the EDA, where compression artifacts are most visible against uniform
# backgrounds).
#
# stddev=0.02 on [0, 1] images ≈ ~5/255 pixel noise — subtle but effective.
# GaussianNoise is only active during training (automatically disabled at
# inference), so it does not affect evaluation or predictions.
augmentation_moderate_noise = Pipeline(
    [
        RandomFlip("horizontal"),
        RandomBrightness(factor=0.08, value_range=(0.0, 1.0)),
        RandomContrast(factor=0.08),
        RandomRotation(factor=0.02, fill_mode="reflect"),
        RandomZoom((-0.05, 0.05), fill_mode="reflect"),
        RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect"
        ),
        GaussianNoise(stddev=0.02),
    ],
    name="augmentation_moderate_noise",
)
