import os
import json
import math
import statistics


def quaternion_to_angle_degrees(q_ref, q_test):
    """
    Compute the *magnitude* of angular difference (in degrees) between two quaternions
    q_ref and q_test. Both q_ref and q_test are [x, y, z, w] floats.

    Steps:
      1) Normalize each quaternion (q_ref, q_test).
      2) dot = dot(q_ref, q_test) (clamped to [-1,1]) => dot = cos(angle/2).
      3) angle = 2 * arccos(dot).    (radians)
      4) angle_deg = angle in degrees.  -> [0..360]
      5) If angle_deg > 180, interpret angle as (360 - angle_deg), so the final
         difference is always in [0..180].
    """

    def normalize_quat(q):
        length = math.sqrt(sum(a*a for a in q))
        if length == 0:
            # fallback if zero-length
            return [0, 0, 0, 1]
        return [a / length for a in q]

    qr = normalize_quat(q_ref)
    qt = normalize_quat(q_test)

    dot_val = sum(a * b for a, b in zip(qr, qt))
    dot_val = max(-1.0, min(1.0, dot_val))  # clamp

    angle_rad = 2.0 * math.acos(dot_val)
    angle_deg = math.degrees(angle_rad)

    # reduce angle > 180 to 360 - angle
    if angle_deg > 180.0:
        angle_deg = 360.0 - angle_deg

    return angle_deg


def translation_distance(t_ref, t_test):
    """
    Euclidean distance between two translations [x, y, z].
    """
    dx = t_test[0] - t_ref[0]
    dy = t_test[1] - t_ref[1]
    dz = t_test[2] - t_ref[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# -----------
# MAIN
# -----------

def main(ref_path, inp_path, details_out, stats_out):
    with open(ref_path, 'r') as f:
        ref_data = json.load(f)
    with open(inp_path, 'r') as f:
        inp_data = json.load(f)

    # Build a quick lookup from the second file by "query"
    inp_lookup = {}
    for item in inp_data:
        inp_lookup[item["query"]] = item

    details = []
    rotation_diffs = []
    translation_diffs = []

    for ref_item in ref_data:
        qid = ref_item["query"]
        if qid not in inp_lookup:
            # If the query doesn't exist in the second file, skip or note missing
            continue

        ref_rot = ref_item["rotation"]
        ref_trn = ref_item["translation"]
        inp_rot = inp_lookup[qid]["rotation"]
        inp_trn = inp_lookup[qid]["translation"]

        # compute diffs
        rot_diff_degs = quaternion_to_angle_degrees(ref_rot, inp_rot)
        trans_diff = translation_distance(ref_trn, inp_trn)

        rotation_diffs.append(rot_diff_degs)
        translation_diffs.append(trans_diff)

        details.append({
            "query": qid,
            "rotation_diff_degs": rot_diff_degs,
            "translation_diff": trans_diff
        })

    # stats
    stat_dict = {}

    def quartiles(values):
        if not values:
            return (None, None, None)
        sorted_vals = sorted(values)
        # Q1, median, Q3
        q1 = statistics.quantiles(sorted_vals, n=4, method='inclusive')[0]
        q2 = statistics.median(sorted_vals)
        q3 = statistics.quantiles(sorted_vals, n=4, method='inclusive')[2]
        return (q1, q2, q3)

    def percentile_10_90(values):
        """
        Return the 10th and 90th percentiles of the given list of values.
        """
        if not values:
            return (None, None)
        sorted_vals = sorted(values)
        # Using statistics.quantiles with n=10 will return 9 cut points:
        #   indices 0..8 correspond to 10%, 20%, ..., 90%.
        p10 = statistics.quantiles(sorted_vals, n=10, method='inclusive')[0]  # 10%
        p90 = statistics.quantiles(sorted_vals, n=10, method='inclusive')[8]  # 90%
        return (p10, p90)

    # rotation stats
    if rotation_diffs:
        stat_dict["rotation_min"] = min(rotation_diffs)
        stat_dict["rotation_max"] = max(rotation_diffs)
        stat_dict["rotation_mean"] = statistics.mean(rotation_diffs)

        (rq1, rq2, rq3) = quartiles(rotation_diffs)
        stat_dict["rotation_q1"] = rq1
        stat_dict["rotation_median"] = rq2
        stat_dict["rotation_q3"] = rq3

        (r10, r90) = percentile_10_90(rotation_diffs)
        stat_dict["rotation_10_percentile"] = r10
        stat_dict["rotation_90_percentile"] = r90
    else:
        stat_dict["rotation_min"] = None
        stat_dict["rotation_max"] = None
        stat_dict["rotation_mean"] = None
        stat_dict["rotation_q1"] = None
        stat_dict["rotation_median"] = None
        stat_dict["rotation_q3"] = None
        stat_dict["rotation_10_percentile"] = None
        stat_dict["rotation_90_percentile"] = None

    # translation stats
    if translation_diffs:
        stat_dict["translation_min"] = min(translation_diffs)
        stat_dict["translation_max"] = max(translation_diffs)
        stat_dict["translation_mean"] = statistics.mean(translation_diffs)

        (tq1, tq2, tq3) = quartiles(translation_diffs)
        stat_dict["translation_q1"] = tq1
        stat_dict["translation_median"] = tq2
        stat_dict["translation_q3"] = tq3

        (t10, t90) = percentile_10_90(translation_diffs)
        stat_dict["translation_10_percentile"] = t10
        stat_dict["translation_90_percentile"] = t90
    else:
        stat_dict["translation_min"] = None
        stat_dict["translation_max"] = None
        stat_dict["translation_mean"] = None
        stat_dict["translation_q1"] = None
        stat_dict["translation_median"] = None
        stat_dict["translation_q3"] = None
        stat_dict["translation_10_percentile"] = None
        stat_dict["translation_90_percentile"] = None

    # write output
    with open(details_out, 'w') as f:
        json.dump(details, f, indent=2)

    with open(stats_out, 'w') as f:
        json.dump(stat_dict, f, indent=2)


if __name__ == "__main__":
    # Example usage:
    gt_folder = "/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb3/hloc/queries_2024"
    query_folder = "/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb3/hloc_2025/queries_2024"
    JSON_FILE_1 = gt_folder + os.sep + "query_results.json"
    JSON_FILE_2 = query_folder + os.sep + "query_results.json"
    OUT_DIFF_FILE = query_folder + os.sep +  "differences.json"
    OUT_STATS_FILE = query_folder + os.sep + "stats.json"

    if os.path.exists(OUT_DIFF_FILE):
        os.remove(OUT_DIFF_FILE)
    if os.path.exists(OUT_STATS_FILE):
        os.remove(OUT_STATS_FILE)

    main(JSON_FILE_1, JSON_FILE_2, OUT_DIFF_FILE, OUT_STATS_FILE)