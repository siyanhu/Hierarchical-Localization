import os
import json
import math
import numpy as np


def quaternion_to_euler_angle(w, x, y, z):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw) in degrees.
    
    Returns (roll, pitch, yaw) in degrees.
    """
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert from radians to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)

    return (roll_deg, pitch_deg, yaw_deg)

def quaternion_diff_angles(q_ref, q_target):
    """
    Compute angle differences (in degrees) between two quaternions:
    - q_ref is the reference quaternion (w, x, y, z)
    - q_target is the quaternion to compare
    
    Returns a dict with roll/pitch/yaw differences (absolute).
    """
    # Convert each to Euler angles
    (r1, p1, y1) = quaternion_to_euler_angle(q_ref[0], q_ref[1], q_ref[2], q_ref[3])
    (r2, p2, y2) = quaternion_to_euler_angle(q_target[0], q_target[1], q_target[2], q_target[3])

    return {
        "roll_diff_deg":  abs(r1 - r2),
        "pitch_diff_deg": abs(p1 - p2),
        "yaw_diff_deg":   abs(y1 - y2)
    }

def translation_distance(t_ref, t_target):
    """
    Euclidean distance between two 3D translation vectors [x, y, z].
    """
    diff = np.array(t_ref) - np.array(t_target)
    return float(np.linalg.norm(diff))

def compute_statistics(values):
    """
    Compute min, max, mean, 25th, 50th (median), and 75th percentile for a list of numeric values.
    Returns a dictionary with those statistics.
    """
    arr = np.array(values)
    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "quartiles": {
                "q1": 0.0,
                "median": 0.0,
                "q3": 0.0
            }
        }
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "quartiles": {
            "q1": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "q3": float(np.percentile(arr, 75))
        }
    }

def compare_against_reference(ref_json, target_json, out_diff_file, out_stats_file):
    """
    Compare one 'target_json' file against the reference 'ref_json'.
    
    Output:
      - A JSON file (out_diff_file) with per-query differences
      - A JSON file (out_stats_file) with overall stats (incl quartiles)
    """
    # Load them
    with open(ref_json, 'r') as f_ref:
        ref_data = json.load(f_ref)
    with open(target_json, 'r') as f_target:
        target_data = json.load(f_target)

    # Turn each into a dict keyed by "query"
    ref_dict = {d["query"]: d for d in ref_data}
    target_dict = {d["query"]: d for d in target_data}

    # We'll accumulate results plus differences for stats
    results = []

    # Accumulators for rotation diffs and translations
    rotation_diffs_all = []  # we'll store all (roll, pitch, yaw) diffs
    translation_diffs_all = []

    # For queries that appear in both
    matched_queries = set(ref_dict.keys()).intersection(set(target_dict.keys()))

    for q_id in sorted(matched_queries):
        ref_item = ref_dict[q_id]
        tgt_item = target_dict[q_id]

        # We assume rotation is [w, x, y, z]. Adjust as needed if your data is in a different order
        rot_ref = ref_item["rotation"]
        rot_tgt = tgt_item["rotation"]

        # We assume translation is [x, y, z]
        tr_ref = ref_item["translation"]
        tr_tgt = tgt_item["translation"]

        # Compute rotation diffs
        rot_diff = quaternion_diff_angles(rot_ref, rot_tgt)
        # We'll store the roll/pitch/yaw diffs for stats
        rotation_diffs_all.extend([rot_diff["roll_diff_deg"],
                                   rot_diff["pitch_diff_deg"],
                                   rot_diff["yaw_diff_deg"]])

        # Compute translation diff
        t_diff = translation_distance(tr_ref, tr_tgt)
        translation_diffs_all.append(t_diff)

        # Store per-query result
        results.append({
            "query": q_id,
            "rotation_diff": rot_diff,
            "translation_diff": t_diff,
            "inference_time_ref": ref_item["inference_time"],
            "inference_time_target": tgt_item["inference_time"]
        })

    # Write the per-query differences
    with open(out_diff_file, "w") as fd:
        json.dump(results, fd, indent=2)

    # Compute stats for rotation and translation
    rotation_stats = compute_statistics(rotation_diffs_all)
    translation_stats = compute_statistics(translation_diffs_all)

    stats_report = {
        "num_matched_queries": len(matched_queries),
        "rotation_diff_degs": rotation_stats,
        "translation_diff": translation_stats
    }

    with open(out_stats_file, "w") as fs:
        json.dump(stats_report, fs, indent=2)

    print(f"Comparison completed. "
          f"Matched queries: {len(matched_queries)}. "
          f"See '{out_diff_file}' and '{out_stats_file}'.")


if __name__ == "__main__":
    # Example usage:
    gt_folder = "/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb3/hloc/queries_2024"
    query_folder = "/media/siyanhu/Changkun/Siyan/Tramway/process/hloc_rgb3_independent_2025/queries_2024"
    JSON_FILE_1 = gt_folder + os.sep + "query_results.json"
    JSON_FILE_2 = query_folder + os.sep + "query_results.json"
    OUT_DIFF_FILE = query_folder + os.sep +  "differences.json"
    OUT_STATS_FILE = query_folder + os.sep + "stats.json"

    compare_against_reference(JSON_FILE_1, JSON_FILE_2, OUT_DIFF_FILE, OUT_STATS_FILE)