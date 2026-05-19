import torch


SK_MAGA_MODALITY = "SK_MAGA_SKELETON"
RGB_SK_MAGA_MODALITY = "RGB_SK_MAGA"

# Internal canonical space:
# 0..21  -> 22 real joints before the final grouped reordering
# 22     -> computed left-hand center
# 23     -> computed right-hand center
NUM_REAL_JOINTS = 22
NUM_MODEL_JOINTS = 24
LEFT_HAND_CENTER_INDEX = 22
RIGHT_HAND_CENTER_INDEX = 23

CANONICAL_24_NAMES = [
    "neck",
    "right_shoulder",
    "nose",
    "left_shoulder",
    "left_eye",
    "right_eye",
    "left_ear",
    "left_elbow",
    "right_elbow",
    "right_ear",
    "left_wrist",
    "left_thumb_tip",
    "left_index_tip",
    "left_middle_tip",
    "left_ring_tip",
    "left_pinky_tip",
    "right_wrist",
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_pinky_tip",
    "left_hand_center",
    "right_hand_center",
]

# 35-point source layout:
# - 0..24  : OpenPose BODY25
# - 25..29 : left-hand thumb/index/middle/ring/pinky fingertips
# - 30..34 : right-hand thumb/index/middle/ring/pinky fingertips
CANONICAL_22_FROM_LIGHT_HAND = [
    1,   # neck
    2,   # right_shoulder
    0,   # nose
    5,   # left_shoulder
    16,  # left_eye
    15,  # right_eye
    18,  # left_ear
    6,   # left_elbow
    3,   # right_elbow
    17,  # right_ear
    7,   # left_wrist
    25,  # left_thumb_tip
    26,  # left_index_tip
    27,  # left_middle_tip
    28,  # left_ring_tip
    29,  # left_pinky_tip
    4,   # right_wrist
    30,  # right_thumb_tip
    31,  # right_index_tip
    32,  # right_middle_tip
    33,  # right_ring_tip
    34,  # right_pinky_tip
]

CANONICAL_P2_LEFT_HAND = [10, 11, 12, 13, 14, 15]
CANONICAL_P3_RIGHT_HAND = [16, 17, 18, 19, 20, 21]
MODEL_24_FROM_CANONICAL_24 = [
    0, 2, 4, 5, 6, 9,
    10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21,
    3, 7, LEFT_HAND_CENTER_INDEX, 1, 8, RIGHT_HAND_CENTER_INDEX,
]


def project_to_canonical_22(raw_points):
    return raw_points[CANONICAL_22_FROM_LIGHT_HAND]


def compute_hand_center(points, point_ids):
    hand_points = points[point_ids]
    valid = ((hand_points[:, 2] > 0) &
             ((hand_points[:, 0] != 0) | (hand_points[:, 1] != 0)))
    if valid.any():
        center_xy = hand_points[valid, :2].mean(dim=0)
        center_score = hand_points[valid, 2].mean()
        return torch.tensor([center_xy[0], center_xy[1], center_score], dtype=torch.float32)
    return torch.zeros(3, dtype=torch.float32)


def append_hand_centers(points_22):
    left_center = compute_hand_center(points_22, CANONICAL_P2_LEFT_HAND)
    right_center = compute_hand_center(points_22, CANONICAL_P3_RIGHT_HAND)
    return torch.cat((points_22, left_center.unsqueeze(0), right_center.unsqueeze(0)), dim=0)


def reorder_canonical_24_to_model_24(points_24):
    return points_24[MODEL_24_FROM_CANONICAL_24]
