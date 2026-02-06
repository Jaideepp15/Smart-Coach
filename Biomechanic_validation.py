import numpy as np
from huggingface_hub import hf_hub_download
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error

from kinematics import Biomechanics

# ============================================================
# CONFIG
# ============================================================

MAX_EPISODES = 200

NOISE_STD_MM = [5, 10, 20]          # Pose perturbation (mm)
SUBSAMPLE_FACTORS = [2, 4]          # Temporal subsampling
SMOOTH = True
SMOOTH_WINDOW = 11
SMOOTH_ORDER = 2

# ============================================================
# LOAD SHOT7M2 FROM HUGGINGFACE
# ============================================================

def load_shot7m2():
    poses_path = hf_hub_download(
        repo_id="amathislab/SHOT7M2",
        repo_type="dataset",
        filename="train/train_dictionary_poses.npy"
    )

    poses_dict = np.load(poses_path, allow_pickle=True).item()
    episodes = poses_dict["sequences"]["keypoints"]

    # Only shooting actions
    shooting_eps = {
        k: v for k, v in episodes.items()
        if k.startswith("casual_player") or k.startswith("intense_player")
    }

    return shooting_eps


def normalize_keypoints(kp):
    """Ensure shape (T, J, 3)."""
    if kp.ndim == 4:
        kp = kp[:, 0]
    return kp


# ============================================================
# UTILS
# ============================================================

def add_noise(kp, std_mm):
    return kp + np.random.normal(0, std_mm / 1000.0, kp.shape)


def compute_rom(x):
    return np.nanmax(x) - np.nanmin(x)


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


# ============================================================
# VALIDATION CORE
# ============================================================

def validate_shot7m2(episodes):
    kin = Biomechanics()

    # --- Oracle biomechanical statistics ---
    stats = {
        "right_elbow_rom": [],
        "left_elbow_rom": [],
        "right_knee_rom": [],
        "left_knee_rom": [],
        "knee_to_elbow_delay": [],
        "elbow_smoothness": [],
        "knee_smoothness": []
    }

    # --- Reconstruction accuracy ---
    angle_mae = {s: [] for s in NOISE_STD_MM}
    rom_error = {s: [] for s in NOISE_STD_MM}

    # --- Temporal stability ---
    temporal_rom_error = []

    for i, (name, kp3d) in enumerate(episodes.items()):
        if i >= MAX_EPISODES:
            break

        kp3d = normalize_keypoints(kp3d)
        if kp3d.ndim != 3 or kp3d.shape[1] < 17:
            continue

        # -------------------------------
        # Oracle biomechanics
        # -------------------------------
        gt_angles = kin.compute_angle_timeseries(
            kp3d, shooting_hand="Right-handed"
        )

        if SMOOTH:
            for j in gt_angles:
                gt_angles[j] = savgol_filter(
                    gt_angles[j], SMOOTH_WINDOW, SMOOTH_ORDER
                )

        # ROM
        stats["right_elbow_rom"].append(compute_rom(gt_angles["right_elbow"]))
        stats["left_elbow_rom"].append(compute_rom(gt_angles["left_elbow"]))
        stats["right_knee_rom"].append(compute_rom(gt_angles["right_knee"]))
        stats["left_knee_rom"].append(compute_rom(gt_angles["left_knee"]))

        # Coordination
        stats["knee_to_elbow_delay"].append(
            np.argmax(gt_angles["right_elbow"]) -
            np.argmax(gt_angles["right_knee"])
        )

        # Smoothness
        stats["elbow_smoothness"].append(
            np.mean(np.abs(np.diff(gt_angles["right_elbow"], 2)))
        )
        stats["knee_smoothness"].append(
            np.mean(np.abs(np.diff(gt_angles["right_knee"], 2)))
        )

        # -------------------------------
        # Noise robustness
        # -------------------------------
        for std in NOISE_STD_MM:
            noisy_kp = add_noise(kp3d, std)
            noisy_angles = kin.compute_angle_timeseries(
                noisy_kp, shooting_hand="Right-handed"
            )

            for joint in gt_angles:
                L = min(len(gt_angles[joint]), len(noisy_angles[joint]))
                if L < 5:
                    continue

                angle_mae[std].append(
                    mean_absolute_error(
                        gt_angles[joint][:L],
                        noisy_angles[joint][:L]
                    )
                )

                rom_error[std].append(
                    abs(
                        compute_rom(noisy_angles[joint][:L]) -
                        compute_rom(gt_angles[joint][:L])
                    )
                )

        # -------------------------------
        # Temporal stability
        # -------------------------------
        for f in SUBSAMPLE_FACTORS:
            sub_kp = kp3d[::f]
            sub_angles = kin.compute_angle_timeseries(
                sub_kp, shooting_hand="Right-handed"
            )

            for joint in gt_angles:
                temporal_rom_error.append(
                    abs(
                        compute_rom(gt_angles[joint]) -
                        compute_rom(sub_angles[joint])
                    )
                )

    return stats, angle_mae, rom_error, temporal_rom_error


# ============================================================
# REPORT
# ============================================================

def summarize(stats, angle_mae, rom_error, temporal_err):
    print("\n=== SHOT7M2 ORACLE BIOMECHANICS ===")
    for k, v in stats.items():
        v = np.array(v)
        print(f"{k}: {v.mean():.3f} ± {v.std():.3f}")

    print("\n=== BIOMECHANICAL RECONSTRUCTION ACCURACY ===")
    for std in angle_mae:
        print(f"\nNoise σ = {std} mm")
        print(
            f"Angle MAE: {np.mean(angle_mae[std]):.3f} ± {np.std(angle_mae[std]):.3f}"
        )
        print(
            f"ROM Error: {np.mean(rom_error[std]):.3f} ± {np.std(rom_error[std]):.3f}"
        )

    print("\nTemporal ROM Stability")
    print(
        f"ΔROM under subsampling: {np.mean(temporal_err):.3f} ± {np.std(temporal_err):.3f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    episodes = load_shot7m2()
    print(f"Total shooting episodes: {len(episodes)}")

    stats, angle_mae, rom_error, temporal_err = validate_shot7m2(episodes)
    summarize(stats, angle_mae, rom_error, temporal_err)


if __name__ == "__main__":
    main()