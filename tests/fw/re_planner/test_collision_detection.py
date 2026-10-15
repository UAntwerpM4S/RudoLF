import math

import numpy as np

from fw.re_planner.collision_detection import CollisionDetector


def test_stationary_dynamic_target_uses_last_heading_for_relative_heading():
    detector = CollisionDetector()
    trajectory = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=float)

    relative_heading = detector._calculate_relative_heading(
        target_ship_velocity=np.zeros(2),
        target_ship_heading=math.pi,
        ship_trajectory=trajectory,
        cpa_point=np.array([50.0, 0.0]),
    )

    assert relative_heading == 0.0


def test_relative_heading_uses_planner_encounter_convention():
    detector = CollisionDetector()
    trajectory = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=float)
    cpa_point = np.array([50.0, 0.0])

    cases = (
        (np.array([1.0, 0.0]), 180.0),   # overtaking / same direction
        (np.array([-1.0, 0.0]), 0.0),    # head-on
        (np.array([0.0, 1.0]), 90.0),    # crossing-right
        (np.array([0.0, -1.0]), 270.0),  # crossing-left
    )

    for velocity, expected in cases:
        assert detector._calculate_relative_heading(
            target_ship_velocity=velocity,
            target_ship_heading=math.atan2(velocity[1], velocity[0]),
            ship_trajectory=trajectory,
            cpa_point=cpa_point,
        ) == expected
