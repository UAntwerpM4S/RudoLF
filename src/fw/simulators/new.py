import numpy as np
import matplotlib.pyplot as plt
import noise
import random
import pandas as pd

# Parameters
width, height = 400, 400
scale = 0.02
width_scale = 0.05
min_width, max_width = 15, 60
point_spacing = 0.5

def generate_river(height, width, scale, width_scale, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    x = random.randint(width // 4, 3 * width // 4)
    bend_offset = random.randint(4000, 6000)
    width_offset = random.randint(0, 10000)

    river_path = []
    for y in range(height):
        bend = noise.pnoise1(y * scale + bend_offset, repeat=9999)
        x += bend * 4
        x = np.clip(x, 0, width - 1)

        w = noise.pnoise1(y * width_scale + width_offset, repeat=9999)
        w = (w + 1) / 2
        river_width = min_width + w * (max_width - min_width)

        

        river_path.append((x, y, river_width))
    return river_path

def get_banks(river_path):
    left_bank, right_bank = [], []
    for (x, y, w) in river_path:
        half_w = w / 2
        left_bank.append((x - half_w, y))
        right_bank.append((x + half_w, y))
    return np.array(left_bank), np.array(right_bank)

def sample_bank_points(bank, spacing=0.5):
    sampled_points = [bank[0]]
    for i in range(1, len(bank)):
        x0, y0 = bank[i-1]
        x1, y1 = bank[i]
        seg_len = np.hypot(x1-x0, y1-y0)
        dx, dy = (x1-x0)/seg_len, (y1-y0)/seg_len
        dist = 0
        while dist < seg_len:
            nx, ny = x0 + dx*dist, y0 + dy*dist
            if np.hypot(nx-sampled_points[-1][0], ny-sampled_points[-1][1]) >= spacing:
                sampled_points.append((nx, ny))
            dist += spacing
    return np.array(sampled_points)

def drop_obstacles(river_path, n_obstacles=5, seed=None):
    """Drop random obstacles inside the river polygon."""
    if seed is not None:
        random.seed(seed)

    obstacles = []
    for _ in range(n_obstacles):
        # Pick a random segment of the river
        x, y, w = random.choice(river_path)

        # Random position within river width
        offset = random.uniform(-w/3, w/3)
        ox = x + offset
        oy = y

        # Random direction (angle in radians)
        direction = random.uniform(0, 2*np.pi)

        # Random radius (could be used for size later)
        radius = random.uniform(0.5, 2.0)

        obstacles.append((ox, oy, direction, radius))

    return obstacles

def export_bank_points(left_pts, right_pts, filename="river_banks.csv"):
    # Create dataframe
    df_left = pd.DataFrame(left_pts, columns=["x", "y"])
    df_left["bank"] = "left"

    df_right = pd.DataFrame(right_pts, columns=["x", "y"])
    df_right["bank"] = "right"

    df = pd.concat([df_left, df_right], ignore_index=True)
    df.to_csv(filename, index=False)
    print(f"✅ Exported river banks to {filename}")

def draw_river_with_obstacles(river_path, left_pts, right_pts, obstacles, terrain):
    # Draw filled river
    left_bank, right_bank = get_banks(river_path)
    xs = np.concatenate([left_bank[:,0], right_bank[::-1,0]])
    ys = np.concatenate([left_bank[:,1], right_bank[::-1,1]])
    plt.fill(xs, ys, color="lightblue", alpha=0.8)

    # Draw dotted banks
    plt.scatter(left_pts[:,0], left_pts[:,1], color="black", s=1)
    plt.scatter(right_pts[:,0], right_pts[:,1], color="black", s=1)

    # Draw obstacles as red points
    ox, oy, _, _ = zip(*obstacles)
    plt.scatter(ox, oy, color="red", s=20, marker="x", label="Obstacles")

    # Terrain background
    plt.imshow(terrain, cmap="terrain", origin="lower", alpha=0.5)

    plt.title("River with Dotted Banks and Obstacles")
    plt.xlim(0, terrain.shape[1])
    plt.ylim(0, terrain.shape[0])
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

# Example usage
seed = None
terrain = np.random.rand(height, width)

river_path = generate_river(height, width, scale, width_scale, seed=seed)
left_bank, right_bank = get_banks(river_path)
left_pts = sample_bank_points(left_bank, spacing=point_spacing)
right_pts = sample_bank_points(right_bank, spacing=point_spacing)

export_bank_points(left_pts, right_pts, "river_banks.csv")

obstacles = drop_obstacles(river_path, n_obstacles=5, seed=seed)

draw_river_with_obstacles(river_path, left_pts, right_pts, obstacles, terrain)
