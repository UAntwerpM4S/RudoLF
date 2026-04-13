"""
Diagnostic script: test whether the ship dynamics + actuator pipeline
produce symmetric responses to positive vs negative rudder commands.

Run from the project root:
    python -m fw.diagnostics.symmetry_test
"""

import numpy as np
from fw.ships.ship_factory import create_ship, ShipModel
from fw.dynamics.fossen import Fossen
from fw.dynamics.nomoto import Nomoto
from fw.simulators.ship_simulator import ShipSimulator
from fw.ships.vessel_state import VesselState


def make_simulator(ship_model=ShipModel.MYZAKO_DEFAULT, dynamics_cls=Fossen, dt=1.0):
    ship = create_ship(ship_model)
    dynamics = dynamics_cls(ship.specifications)
    return ShipSimulator(ship.specifications, dynamics, dt, wind=False, current=False, numerical_damping=False)


def reset_sim(sim, heading=0.0, u=2.0):
    sim.reset(VesselState(
        terminated=False, truncated=False,
        x=0.0, y=0.0, heading=heading,
        u=u, v=0.0, r=0.0,
    ))


def test_dynamics_raw_symmetry():
    """Test 1: Raw dynamics accelerations for +/- rudder at identical state."""
    print("=" * 70)
    print("TEST 1: Raw Fossen.accelerations() symmetry")
    print("=" * 70)

    ship = create_ship(ShipModel.MYZAKO_DEFAULT)
    dyn = Fossen(ship.specifications)

    # Test at several operating points
    test_states = [
        {"u": 2.0, "v": 0.0, "r": 0.0, "thrust": 0.5, "label": "straight, u=2"},
        {"u": 2.0, "v": 0.3, "r": 0.1, "thrust": 0.5, "label": "with sway & yaw"},
        {"u": 0.5, "v": 0.0, "r": 0.0, "thrust": 0.5, "label": "low speed"},
        {"u": 4.0, "v": 0.0, "r": 0.0, "thrust": 1.0, "label": "high speed"},
    ]

    for state in test_states:
        u, v, r, thrust = state["u"], state["v"], state["r"], state["thrust"]

        du_pos, dv_pos, dr_pos = dyn.accelerations(u, v, r, rudder_angle=+0.5, thrust=thrust)
        du_neg, dv_neg, dr_neg = dyn.accelerations(u, v, r, rudder_angle=-0.5, thrust=thrust)

        print(f"\n  State: {state['label']}")
        print(f"    +rudder: du={du_pos:+.6e}  dv={dv_pos:+.6e}  dr={dr_pos:+.6e}")
        print(f"    -rudder: du={du_neg:+.6e}  dv={dv_neg:+.6e}  dr={dr_neg:+.6e}")
        print(f"    du symmetric: {np.isclose(du_pos, du_neg)}  (diff={du_pos - du_neg:+.2e})")
        print(f"    dv symmetric: {np.isclose(dv_pos, -dv_neg)}  (diff={dv_pos + dv_neg:+.2e})")
        print(f"    dr symmetric: {np.isclose(dr_pos, -dr_neg)}  (diff={dr_pos + dr_neg:+.2e})")

    # Also test with nonzero v,r to check cross-coupling asymmetry
    print(f"\n  --- Cross-coupling check (v != 0) ---")
    for v_test in [0.0, 0.5, -0.5]:
        du_p, dv_p, dr_p = dyn.accelerations(2.0, v_test, 0.0, +0.5, 0.5)
        du_n, dv_n, dr_n = dyn.accelerations(2.0, v_test, 0.0, -0.5, 0.5)
        print(f"    v={v_test:+.1f}: dr_pos={dr_p:+.6e}, dr_neg={dr_n:+.6e}, "
              f"sum={dr_p + dr_n:+.2e} (0 = symmetric)")


def test_simulator_symmetry():
    """Test 2: Full simulator pipeline — step with +/- rudder, compare trajectories."""
    print("\n" + "=" * 70)
    print("TEST 2: Full simulator step() symmetry (with actuator pipeline)")
    print("=" * 70)

    n_steps = 200

    for smoothing in [False, True]:
        sim_pos = make_simulator()
        sim_neg = make_simulator()
        reset_sim(sim_pos, heading=0.0, u=2.0)
        reset_sim(sim_neg, heading=0.0, u=2.0)

        action_pos = np.array([+1.0, 0.5], dtype=np.float32)  # full starboard rudder
        action_neg = np.array([-1.0, 0.5], dtype=np.float32)  # full port rudder

        print(f"\n  Smoothing={'ON' if smoothing else 'OFF'}, {n_steps} steps, dt=1.0s")
        print(f"  {'step':>5s} | {'x_pos':>10s} {'y_pos':>10s} {'h_pos':>10s} {'r_pos':>10s} | "
              f"{'x_neg':>10s} {'y_neg':>10s} {'h_neg':>10s} {'r_neg':>10s} | "
              f"{'dy_sym':>10s} {'dr_sym':>10s}")

        for step in range(n_steps):
            sim_pos.step(action_pos, enable_smoothing=smoothing)
            sim_neg.step(action_neg, enable_smoothing=smoothing)

            sp = sim_pos.state
            sn = sim_neg.state

            # Perfect symmetry: x should match, y should be opposite, heading opposite, r opposite
            dy_sym_err = sp.y + sn.y  # should be 0
            dr_sym_err = sp.r + sn.r  # should be 0

            if step % 20 == 0 or step == n_steps - 1:
                print(f"  {step:5d} | {sp.x:10.4f} {sp.y:10.4f} {sp.heading:10.6f} {sp.r:10.6f} | "
                      f"{sn.x:10.4f} {sn.y:10.4f} {sn.heading:10.6f} {sn.r:10.6f} | "
                      f"{dy_sym_err:+10.2e} {dr_sym_err:+10.2e}")

        # Final summary
        sp = sim_pos.state
        sn = sim_neg.state
        print(f"\n  Final state comparison (smoothing={'ON' if smoothing else 'OFF'}):")
        print(f"    x:  pos={sp.x:.6f}, neg={sn.x:.6f}, diff={sp.x - sn.x:+.2e}")
        print(f"    y:  pos={sp.y:.6f}, neg={sn.y:.6f}, sum={sp.y + sn.y:+.2e} (0=symmetric)")
        print(f"    heading: pos={sp.heading:.6f}, neg={sn.heading:.6f}, sum={sp.heading + sn.heading:+.2e}")
        print(f"    u:  pos={sp.u:.6f}, neg={sn.u:.6f}, diff={sp.u - sn.u:+.2e}")
        print(f"    v:  pos={sp.v:.6f}, neg={sn.v:.6f}, sum={sp.v + sn.v:+.2e}")
        print(f"    r:  pos={sp.r:.6f}, neg={sn.r:.6f}, sum={sp.r + sn.r:+.2e}")


def test_observation_symmetry():
    """Test 3: Check if the polar scan sector ordering is symmetric."""
    print("\n" + "=" * 70)
    print("TEST 3: Polar scan sector angle ordering")
    print("=" * 70)

    n_sectors = 54
    fov = np.radians(270)
    bin_width = fov / n_sectors
    rel_angles = -(fov / 2.0) + (np.arange(n_sectors, dtype=np.float32) + 0.5) * bin_width

    print(f"  FOV: {np.degrees(fov):.1f} deg, {n_sectors} sectors")
    print(f"  Angle range: [{np.degrees(rel_angles[0]):.2f}, {np.degrees(rel_angles[-1]):.2f}] deg")
    print(f"  Center angle: {np.degrees(rel_angles[n_sectors // 2]):.2f} deg (sector {n_sectors // 2})")

    # Check if angles are symmetric around 0
    is_symmetric = np.allclose(rel_angles, -rel_angles[::-1])
    print(f"  Angles symmetric around 0: {is_symmetric}")

    if not is_symmetric:
        print("  WARNING: Sector angles are NOT symmetric!")
        for i in range(n_sectors // 2):
            j = n_sectors - 1 - i
            print(f"    sector[{i}]={np.degrees(rel_angles[i]):+.2f} vs "
                  f"-sector[{j}]={-np.degrees(rel_angles[j]):+.2f}, "
                  f"diff={np.degrees(rel_angles[i] + rel_angles[j]):+.4f}")

    # Check if the dead-ahead direction (0 deg) falls on a sector boundary or center
    closest_to_zero = np.argmin(np.abs(rel_angles))
    print(f"  Closest sector to dead-ahead: index {closest_to_zero}, "
          f"angle={np.degrees(rel_angles[closest_to_zero]):+.2f} deg")
    print(f"  Even sector count: {n_sectors % 2 == 0}")
    if n_sectors % 2 == 0:
        print(f"  NOTE: Even sector count means dead-ahead falls on a sector BOUNDARY,")
        print(f"        not a sector center. The two sectors straddling 0 deg are:")
        print(f"        sector[{n_sectors//2 - 1}] = {np.degrees(rel_angles[n_sectors//2 - 1]):+.2f} deg")
        print(f"        sector[{n_sectors//2}] = {np.degrees(rel_angles[n_sectors//2]):+.2f} deg")


def test_velocity_clipping_symmetry():
    """Test 4: Check if surge_limits clipping creates asymmetry."""
    print("\n" + "=" * 70)
    print("TEST 4: Velocity limit clipping asymmetry check")
    print("=" * 70)

    ship = create_ship(ShipModel.MYZAKO_DEFAULT)
    specs = ship.specifications
    print(f"  surge_limits: {specs.surge_limits}")
    print(f"  sway_limits:  {specs.sway_limits}")
    print(f"  yaw_rate_limits: {specs.yaw_rate_limits}")

    # Surge is [0, 5] — NOT symmetric around 0. This means the ship can't go backward.
    # This shouldn't cause left/right asymmetry, but worth noting.
    if specs.surge_limits[0] != -specs.surge_limits[1]:
        print(f"  NOTE: surge_limits are NOT symmetric around 0 "
              f"({specs.surge_limits[0]} != {-specs.surge_limits[1]})")
        print(f"        This is expected (no reverse) but doesn't affect left/right symmetry.")

    # Sway and yaw should be symmetric
    sway_sym = np.isclose(specs.sway_limits[0], -specs.sway_limits[1])
    yaw_sym = np.isclose(specs.yaw_rate_limits[0], -specs.yaw_rate_limits[1])
    print(f"  sway_limits symmetric: {sway_sym}")
    print(f"  yaw_rate_limits symmetric: {yaw_sym}")
    if not sway_sym:
        print(f"  WARNING: Sway limits are asymmetric! This could cause left/right bias.")
    if not yaw_sym:
        print(f"  WARNING: Yaw rate limits are asymmetric! This could cause left/right bias.")


def test_heading_dependent_asymmetry():
    """Test 5: Check if turning from a non-zero heading produces asymmetric results
    due to floating-point or wrapping effects."""
    print("\n" + "=" * 70)
    print("TEST 5: Heading-dependent asymmetry (wrapping effects)")
    print("=" * 70)

    headings_to_test = [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2, 3.0]
    n_steps = 100

    for h0 in headings_to_test:
        sim_pos = make_simulator()
        sim_neg = make_simulator()
        reset_sim(sim_pos, heading=h0, u=2.0)
        reset_sim(sim_neg, heading=h0, u=2.0)

        for _ in range(n_steps):
            sim_pos.step(np.array([+1.0, 0.5], dtype=np.float32), enable_smoothing=False)
            sim_neg.step(np.array([-1.0, 0.5], dtype=np.float32), enable_smoothing=False)

        sp = sim_pos.state
        sn = sim_neg.state

        # For symmetric dynamics, the yaw rate magnitude should be equal
        r_diff = abs(sp.r) - abs(sn.r)
        # The displacement perpendicular to initial heading should be equal magnitude
        # Project final position onto perpendicular axis of initial heading
        perp_pos = -np.sin(h0) * sp.x + np.cos(h0) * sp.y
        perp_neg = -np.sin(h0) * sn.x + np.cos(h0) * sn.y

        print(f"  h0={np.degrees(h0):+7.1f} deg: |r_pos|={abs(sp.r):.6f}, |r_neg|={abs(sn.r):.6f}, "
              f"diff={r_diff:+.2e}, "
              f"perp_pos={perp_pos:+.4f}, perp_neg={perp_neg:+.4f}, "
              f"sum={perp_pos + perp_neg:+.2e}")


if __name__ == "__main__":
    test_dynamics_raw_symmetry()
    test_simulator_symmetry()
    test_observation_symmetry()
    test_velocity_clipping_symmetry()
    test_heading_dependent_asymmetry()

    print("\n" + "=" * 70)
    print("DONE. If all 'sum' and 'diff' values above are near zero (~1e-6 or less),")
    print("then the dynamics are symmetric and the left-turn bias is NOT caused by")
    print("the physics engine. Look elsewhere (path geometry, observation encoding,")
    print("or NN initialization/training dynamics).")
    print("=" * 70)
