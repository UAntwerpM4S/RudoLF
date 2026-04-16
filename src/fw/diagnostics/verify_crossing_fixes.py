"""
Quick verification of the two crossing-scenario fixes:
1. Uniform segment selection (no length bias)
2. Activation timing cap for narrow encounters
"""
import numpy as np

# ============================================================
# 1. Segment selection: uniform vs length-weighted
# ============================================================
rng = np.random.default_rng(42)

# Simulate 5 segments with varying lengths (narrow canal = short segments)
seg_lengths = np.array([50.0, 120.0, 300.0, 80.0, 450.0])
seg_names = ["narrow-50m", "med-120m", "wide-300m", "med-80m", "wide-450m"]
n_segments = len(seg_lengths)
n_trials = 50_000

# OLD: length-weighted selection (first segment tried is almost always the longest)
first_chosen_old = np.zeros(n_segments, dtype=int)
for _ in range(n_trials):
    probs = seg_lengths / seg_lengths.sum()
    order = rng.choice(n_segments, size=n_segments, replace=False, p=probs)
    first_chosen_old[order[0]] += 1

# NEW: uniform selection
first_chosen_new = np.zeros(n_segments, dtype=int)
for _ in range(n_trials):
    order = rng.choice(n_segments, size=n_segments, replace=False)
    first_chosen_new[order[0]] += 1

print("=" * 65)
print("SEGMENT SELECTION — first segment tried (out of {:,} trials)".format(n_trials))
print("=" * 65)
print(f"{'Segment':<14} {'Length':>6}  {'OLD (len-weighted)':>20}  {'NEW (uniform)':>20}")
print("-" * 65)
for i in range(n_segments):
    pct_old = 100 * first_chosen_old[i] / n_trials
    pct_new = 100 * first_chosen_new[i] / n_trials
    print(f"{seg_names[i]:<14} {seg_lengths[i]:>5.0f}m  {first_chosen_old[i]:>8} ({pct_old:>5.1f}%)  {first_chosen_new[i]:>8} ({pct_new:>5.1f}%)")
print()

# ============================================================
# 2. Activation timing: old vs new
# ============================================================
print("=" * 65)
print("ACTIVATION TIMING — arrival time comparison")
print("=" * 65)

# Parameters (match codebase defaults)
activation_range_base = 550.0  # polar_max_range(500) + margin(50)
v_os = 2.0                     # typical OS speed
speed_factor = 1.0             # crossing_speed_factor
v_min = 1.0                    # crossing_ts_v_min
v_max = 5.0                    # crossing_ts_v_max

d_ts_values = [250, 150, 100, 80, 50]

print(f"{'d_ts':>6}  {'d_os_old':>8}  {'v_ts_old':>8}  {'t_ts':>6}  {'t_os':>6}  {'delta':>7}  |  {'d_os_new':>8}  {'v_ts_new':>8}  {'t_ts':>6}  {'t_os':>6}  {'delta':>7}")
print("-" * 110)

for d_ts in d_ts_values:
    # OLD: always activate at full range, clamp speed
    d_os_old = activation_range_base
    v_ts_raw = speed_factor * d_ts * v_os / d_os_old
    v_ts_old = float(np.clip(v_ts_raw, v_min, v_max))
    t_ts_old = d_ts / v_ts_old
    t_os_old = d_os_old / v_os
    delta_old = t_ts_old - t_os_old

    # NEW: cap activation distance
    timing_cap = speed_factor * d_ts * v_os / v_min
    d_os_new = min(activation_range_base, timing_cap)
    v_ts_raw_new = speed_factor * d_ts * v_os / d_os_new
    v_ts_new = float(np.clip(v_ts_raw_new, v_min, v_max))
    t_ts_new = d_ts / v_ts_new
    t_os_new = d_os_new / v_os
    delta_new = t_ts_new - t_os_new

    print(
        f"{d_ts:>5}m  {d_os_old:>7.0f}m  {v_ts_old:>7.2f}  {t_ts_old:>5.0f}s  {t_os_old:>5.0f}s  {delta_old:>+6.0f}s  |  "
        f"{d_os_new:>7.0f}m  {v_ts_new:>7.2f}  {t_ts_new:>5.0f}s  {t_os_new:>5.0f}s  {delta_new:>+6.0f}s"
    )

print()
print("delta = t_ts - t_os  (negative = TS arrives before OS)")
print("With speed_factor=1.0, the NEW timing gives simultaneous arrival (delta=0).")
print("With speed_factor>1.0, TS intentionally arrives before OS (forcing reaction).")
