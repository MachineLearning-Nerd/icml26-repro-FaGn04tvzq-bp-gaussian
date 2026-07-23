"""Real Middlebury Cones audit for Claim 6 of arXiv:2601.21935.

Only NumPy and the Python standard library are used.  The official 2003 Cones
stereo pair is downloaded as PPM/PGM, resized to the paper's 150x200 grid, and
solved with synchronous non-parametric BP and Gaussian BP on the same masked
four-neighbour factor graph.
"""
from __future__ import annotations

from io import BytesIO
import json
import os
import time
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


DATA_URL = (
    "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/"
    "cones-ppm-2.zip"
)
HEIGHT = 150
WIDTH = 200
PATCH_SIZE = 5
LAMBDA = 0.002
EDGE_THRESHOLD = 3.0
MAX_ITERATIONS = 2000
SEEDS = tuple(range(42, 47))
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = (1, 0, 3, 2)
TINY32 = np.float32(1e-30)


def _read_pnm(payload: bytes) -> np.ndarray:
    position = 0

    def token() -> bytes:
        nonlocal position
        while position < len(payload):
            if payload[position : position + 1] == b"#":
                position = payload.find(b"\n", position) + 1
            elif payload[position] in b" \t\r\n":
                position += 1
            else:
                break
        start = position
        while position < len(payload) and payload[position] not in b" \t\r\n":
            position += 1
        return payload[start:position]

    magic = token()
    width = int(token())
    height = int(token())
    maximum = int(token())
    if magic not in (b"P5", b"P6") or maximum != 255:
        raise ValueError(f"unsupported PNM header: {magic!r}, max={maximum}")
    while position < len(payload) and payload[position] in b" \t\r\n":
        position += 1
    channels = 3 if magic == b"P6" else 1
    values = np.frombuffer(payload, dtype=np.uint8, count=height * width * channels, offset=position)
    shape = (height, width, channels) if channels == 3 else (height, width)
    return values.reshape(shape).copy()


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    y = np.linspace(0.0, source_h - 1.0, height)
    x = np.linspace(0.0, source_w - 1.0, width)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, source_h - 1)
    x1 = np.minimum(x0 + 1, source_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    trailing = (1,) * (image.ndim - 2)
    wy = wy.reshape((height, 1) + trailing)
    wx = wx.reshape((1, width) + trailing)
    top = image[y0[:, None], x0[None, :]].astype(np.float32) * (1.0 - wx)
    top += image[y0[:, None], x1[None, :]].astype(np.float32) * wx
    bottom = image[y1[:, None], x0[None, :]].astype(np.float32) * (1.0 - wx)
    bottom += image[y1[:, None], x1[None, :]].astype(np.float32) * wx
    return top * (1.0 - wy) + bottom * wy


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    y = np.rint(np.linspace(0.0, image.shape[0] - 1.0, height)).astype(int)
    x = np.rint(np.linspace(0.0, image.shape[1] - 1.0, width)).astype(int)
    return image[y[:, None], x[None, :]].copy()


def load_cones() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with urlopen(DATA_URL, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        left_raw = _read_pnm(archive.read("cones/im2.ppm"))
        right_raw = _read_pnm(archive.read("cones/im6.ppm"))
        disparity_raw = _read_pnm(archive.read("cones/disp2.pgm"))

    left = _resize_bilinear(left_raw, HEIGHT, WIDTH)
    right = _resize_bilinear(right_raw, HEIGHT, WIDTH)
    disparity = _resize_nearest(disparity_raw, HEIGHT, WIDTH).astype(np.float32) / 4.0
    horizontal_scale = WIDTH / float(disparity_raw.shape[1])
    disparity *= horizontal_scale
    metadata = {
        "url": DATA_URL,
        "original_shape": list(left_raw.shape[:2]),
        "experiment_shape": [HEIGHT, WIDTH],
        "variables": HEIGHT * WIDTH,
        "horizontal_disparity_scale": horizontal_scale,
        "archive_bytes": len(archive_bytes),
    }
    return left, right, disparity, metadata


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def photometric_prior(
    left: np.ndarray, right: np.ndarray, disparities: int
) -> tuple[np.ndarray, np.ndarray]:
    left_gray = np.mean(left, axis=2, dtype=np.float32)
    right_gray = np.mean(right, axis=2, dtype=np.float32)
    costs = np.empty((HEIGHT, WIDTH, disparities), dtype=np.float32)
    radius = PATCH_SIZE // 2
    invalid_cost = np.float32(255.0**2)
    for disparity in range(disparities):
        squared = np.full((HEIGHT, WIDTH), invalid_cost, dtype=np.float32)
        if disparity == 0:
            delta = left_gray - right_gray
            squared[:] = delta * delta
        else:
            delta = left_gray[:, disparity:] - right_gray[:, :-disparity]
            squared[:, disparity:] = delta * delta
        costs[:, :, disparity] = _box_mean(squared, radius)
    costs -= np.min(costs, axis=2, keepdims=True)
    log_prior = -np.float32(LAMBDA) * costs
    log_prior -= np.max(log_prior, axis=2, keepdims=True)
    prior = np.exp(log_prior).astype(np.float32)
    prior /= np.sum(prior, axis=2, keepdims=True)
    return prior, left_gray


def edge_activity(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.zeros((4, HEIGHT, WIDTH), dtype=bool)
    vertical = np.abs(gray[1:, :] - gray[:-1, :]) <= EDGE_THRESHOLD
    horizontal = np.abs(gray[:, 1:] - gray[:, :-1]) <= EDGE_THRESHOLD
    active[0, 1:, :] = vertical
    active[1, :-1, :] = vertical
    active[2, :, 1:] = horizontal
    active[3, :, :-1] = horizontal
    edge_pixel = np.zeros((HEIGHT, WIDTH), dtype=bool)
    edge_pixel[1:, :] |= ~vertical
    edge_pixel[:-1, :] |= ~vertical
    edge_pixel[:, 1:] |= ~horizontal
    edge_pixel[:, :-1] |= ~horizontal
    return active, edge_pixel


def _slices(direction: int):
    if direction == 0:
        return (slice(1, None), slice(None)), (slice(None, -1), slice(None))
    if direction == 1:
        return (slice(None, -1), slice(None)), (slice(1, None), slice(None))
    if direction == 2:
        return (slice(None), slice(1, None)), (slice(None), slice(None, -1))
    return (slice(None), slice(None, -1)), (slice(None), slice(1, None))


def _softmax(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - np.max(log_values, axis=-1, keepdims=True)
    values = np.exp(shifted).astype(np.float32)
    values /= np.sum(values, axis=-1, keepdims=True)
    return values


def _convolve_labels(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    center = kernel.size // 2
    for index, weight in enumerate(kernel):
        shift = index - center
        if shift < 0:
            result[..., :shift] += weight * values[..., -shift:]
        elif shift > 0:
            result[..., shift:] += weight * values[..., :-shift]
        else:
            result += weight * values
    result = np.maximum(result, TINY32)
    result /= np.sum(result, axis=-1, keepdims=True)
    return result


def _prediction_mse(belief: np.ndarray, labels: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.sum(belief * labels, axis=2)
    valid = truth > 0.0
    return float(np.mean((prediction[valid] - truth[valid]) ** 2))


def nonparametric_bp(
    prior: np.ndarray,
    active: np.ndarray,
    truth: np.ndarray,
    seed: int,
    pairwise_kernel: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    disparities = prior.shape[2]
    labels = np.arange(disparities, dtype=np.float32)
    rng = np.random.default_rng(seed)
    messages = np.full((4, HEIGHT, WIDTH, disparities), 1.0 / disparities, dtype=np.float32)
    messages *= 1.0 + rng.uniform(-1e-3, 1e-3, size=messages.shape).astype(np.float32)
    messages /= np.sum(messages, axis=3, keepdims=True)
    log_prior = np.log(np.maximum(prior, TINY32))
    stable_checks = 0
    previous_mse = np.inf
    trajectory = []
    damping = np.float32(0.5)
    uniform = np.float32(1.0 / disparities)

    for iteration in range(1, MAX_ITERATIONS + 1):
        log_messages = np.log(np.maximum(messages, TINY32))
        total = log_prior + np.sum(log_messages, axis=0)
        updated = np.full_like(messages, uniform)
        for direction in range(4):
            source_slice, target_slice = _slices(direction)
            cavity_log = total[source_slice] - log_messages[direction][source_slice]
            cavity = _softmax(cavity_log)
            outgoing = _convolve_labels(cavity, pairwise_kernel)
            enabled = active[direction][source_slice][..., None]
            outgoing = np.where(enabled, outgoing, uniform)
            updated[(OPPOSITE[direction], *target_slice)] = outgoing
        messages = damping * updated + (1.0 - damping) * messages

        if iteration % 20 == 0 or iteration == 1:
            belief = _softmax(log_prior + np.sum(np.log(np.maximum(messages, TINY32)), axis=0))
            mse = _prediction_mse(belief, labels, truth)
            trajectory.append({"iteration": iteration, "mse": mse})
            print(f"    BP seed={seed} iteration={iteration} MSE={mse:.6f}")
            if iteration >= 100 and abs(previous_mse - mse) < 1e-5:
                stable_checks += 1
            else:
                stable_checks = 0
            previous_mse = mse
            if stable_checks >= 5:
                break

    belief = _softmax(log_prior + np.sum(np.log(np.maximum(messages, TINY32)), axis=0))
    return belief, {
        "seed": seed,
        "iterations_executed": iteration,
        "max_iterations": MAX_ITERATIONS,
        "final_mse": _prediction_mse(belief, labels, truth),
        "trajectory": trajectory,
    }


def gaussian_bp(
    prior: np.ndarray,
    active: np.ndarray,
    truth: np.ndarray,
    seed: int,
    pairwise_variance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    disparities = prior.shape[2]
    labels = np.arange(disparities, dtype=np.float32)
    # A global moment fit can put the Gaussian between distinct photometric
    # modes, where the unary factor has little support.  GBP instead uses a
    # local Laplace projection around the dominant disparity mode.  A flat
    # mode retains the variance of a discrete uniform prior, while curvature
    # supplies precision when the photometric match is informative.
    log_prior = np.log(np.maximum(prior, TINY32))
    mode = np.argmax(prior, axis=2)
    center = np.take_along_axis(log_prior, mode[:, :, None], axis=2)[:, :, 0]
    lower_index = np.maximum(mode - 1, 0)
    upper_index = np.minimum(mode + 1, disparities - 1)
    lower = np.take_along_axis(log_prior, lower_index[:, :, None], axis=2)[:, :, 0]
    upper = np.take_along_axis(log_prior, upper_index[:, :, None], axis=2)[:, :, 0]
    curvature = 2.0 * center - lower - upper
    one_sided = np.where(mode == 0, 2.0 * (center - upper), curvature)
    one_sided = np.where(mode == disparities - 1, 2.0 * (center - lower), one_sided)
    uniform_variance = np.float32((disparities**2 - 1) / 12.0)
    minimum_precision = np.float32(1.0 / uniform_variance)
    prior_precision = np.maximum(one_sided, minimum_precision).astype(np.float32)
    prior_variance = np.clip(1.0 / prior_precision, 0.25, uniform_variance)
    subpixel_offset = 0.5 * (upper - lower) / np.maximum(prior_precision, np.float32(1e-8))
    subpixel_offset = np.clip(subpixel_offset, -0.5, 0.5)
    subpixel_offset = np.where((mode == 0) | (mode == disparities - 1), 0.0, subpixel_offset)
    prior_mean = np.clip(mode.astype(np.float32) + subpixel_offset, 0.0, disparities - 1.0)
    prior_precision = 1.0 / prior_variance
    prior_information = prior_mean * prior_precision

    rng = np.random.default_rng(seed)
    message_precision = rng.uniform(0.0, 1e-5, size=(4, HEIGHT, WIDTH)).astype(np.float32)
    message_information = message_precision * prior_mean[None, :, :]
    damping = np.float32(0.5)
    stable_checks = 0
    previous_mse = np.inf
    trajectory = []
    valid = truth > 0.0

    for iteration in range(1, MAX_ITERATIONS + 1):
        total_precision = prior_precision + np.sum(message_precision, axis=0)
        total_information = prior_information + np.sum(message_information, axis=0)
        updated_precision = np.zeros_like(message_precision)
        updated_information = np.zeros_like(message_information)
        for direction in range(4):
            source_slice, target_slice = _slices(direction)
            cavity_precision = total_precision[source_slice] - message_precision[direction][source_slice]
            cavity_information = total_information[source_slice] - message_information[direction][source_slice]
            cavity_precision = np.maximum(cavity_precision, np.float32(1e-8))
            outgoing_mean = cavity_information / cavity_precision
            outgoing_precision = 1.0 / (1.0 / cavity_precision + pairwise_variance)
            enabled = active[direction][source_slice]
            outgoing_precision = np.where(enabled, outgoing_precision, 0.0)
            outgoing_information = outgoing_mean * outgoing_precision
            updated_precision[(OPPOSITE[direction], *target_slice)] = outgoing_precision
            updated_information[(OPPOSITE[direction], *target_slice)] = outgoing_information
        message_precision = damping * updated_precision + (1.0 - damping) * message_precision
        message_information = damping * updated_information + (1.0 - damping) * message_information

        if iteration % 20 == 0 or iteration == 1:
            precision = prior_precision + np.sum(message_precision, axis=0)
            information = prior_information + np.sum(message_information, axis=0)
            prediction = information / precision
            mse = float(np.mean((prediction[valid] - truth[valid]) ** 2))
            trajectory.append({"iteration": iteration, "mse": mse})
            print(f"    GBP seed={seed} iteration={iteration} MSE={mse:.6f}")
            if iteration >= 100 and abs(previous_mse - mse) < 1e-5:
                stable_checks += 1
            else:
                stable_checks = 0
            previous_mse = mse
            if stable_checks >= 5:
                break

    precision = prior_precision + np.sum(message_precision, axis=0)
    information = prior_information + np.sum(message_information, axis=0)
    mean = information / precision
    variance = 1.0 / precision
    gaussian = np.exp(
        -0.5 * (labels[None, None, :] - mean[:, :, None]) ** 2 / variance[:, :, None]
    ).astype(np.float32)
    gaussian /= np.sum(gaussian, axis=2, keepdims=True)
    return gaussian, {
        "seed": seed,
        "iterations_executed": iteration,
        "max_iterations": MAX_ITERATIONS,
        "final_mse": float(np.mean((mean[valid] - truth[valid]) ** 2)),
        "trajectory": trajectory,
    }


def belief_kl_map(belief: np.ndarray) -> np.ndarray:
    labels = np.arange(belief.shape[2], dtype=np.float32)
    mean = np.sum(belief * labels, axis=2)
    variance = np.sum(belief * (labels[None, None, :] - mean[:, :, None]) ** 2, axis=2)
    variance = np.maximum(variance, np.float32(0.25))
    gaussian = np.exp(
        -0.5 * (labels[None, None, :] - mean[:, :, None]) ** 2 / variance[:, :, None]
    ).astype(np.float32)
    gaussian /= np.sum(gaussian, axis=2, keepdims=True)
    return np.sum(
        belief * (np.log(np.maximum(belief, TINY32)) - np.log(np.maximum(gaussian, TINY32))),
        axis=2,
    )


def _region_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = values[mask]
    return {
        "pixels": int(selected.size),
        "mean_kl": float(np.mean(selected)),
        "median_kl": float(np.median(selected)),
        "fraction_below_0.02": float(np.mean(selected < 0.02)),
    }


def run_stereo_audit(output_directory: str) -> dict[str, object]:
    started = time.perf_counter()
    left, right, truth, metadata = load_cones()
    maximum_truth = float(np.max(truth))
    disparities = max(16, int(np.ceil(maximum_truth)) + 3)
    prior, gray = photometric_prior(left, right, disparities)
    active, edge_pixels = edge_activity(gray)
    offsets = np.arange(-4, 5, dtype=np.float32)
    pairwise_kernel = np.exp(-0.5 * (offsets / 1.5) ** 2).astype(np.float32)
    pairwise_kernel /= np.sum(pairwise_kernel)
    pairwise_variance = float(np.sum(pairwise_kernel * offsets**2))
    labels = np.arange(disparities, dtype=np.float32)
    prior_mean = np.sum(prior * labels, axis=2)
    prior_variance = np.sum(prior * (labels[None, None, :] - prior_mean[:, :, None]) ** 2, axis=2)
    weak_prior = prior_variance >= np.quantile(prior_variance, 0.75)
    strong_prior = prior_variance <= np.quantile(prior_variance, 0.25)

    print(f"  source={DATA_URL}")
    print(f"  graph={HEIGHT}x{WIDTH} ({HEIGHT * WIDTH} variables), labels={disparities}, patch={PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  lambda={LAMBDA}, edge_threshold={EDGE_THRESHOLD}, max_iterations={MAX_ITERATIONS}, seeds={list(SEEDS)}")
    print(f"  valid_ground_truth_pixels={int(np.sum(truth > 0.0))}, maximum_scaled_disparity={maximum_truth:.3f}")
    print(f"  active_directed_edge_fraction={float(np.mean(active)):.6f}")

    bp_runs = []
    gbp_runs = []
    representative_belief = None
    for seed in SEEDS:
        belief, bp_record = nonparametric_bp(prior, active, truth, seed, pairwise_kernel)
        gaussian, gbp_record = gaussian_bp(prior, active, truth, seed, pairwise_variance)
        bp_runs.append(bp_record)
        gbp_runs.append(gbp_record)
        if representative_belief is None:
            representative_belief = belief

    bp_mses = np.asarray([record["final_mse"] for record in bp_runs])
    gbp_mses = np.asarray([record["final_mse"] for record in gbp_runs])
    mse_gap = float(abs(np.mean(bp_mses) - np.mean(gbp_mses)))
    relative_gap = mse_gap / max(float(np.mean(bp_mses)), 1e-12)
    kl_map = belief_kl_map(representative_belief)
    regions = {
        "weak_prior": _region_summary(kl_map, weak_prior),
        "strong_prior": _region_summary(kl_map, strong_prior),
        "low_contrast": _region_summary(kl_map, ~edge_pixels),
        "high_contrast_edge": _region_summary(kl_map, edge_pixels),
    }
    mse_aligned = relative_gap < 0.10
    spatial_aligned = (
        regions["low_contrast"]["mean_kl"] < regions["high_contrast_edge"]["mean_kl"]
        and regions["low_contrast"]["fraction_below_0.02"]
        > regions["high_contrast_edge"]["fraction_below_0.02"]
    )
    result = {
        "paper_settings": {
            "image_shape": [HEIGHT, WIDTH],
            "variables": HEIGHT * WIDTH,
            "patch_size": PATCH_SIZE,
            "lambda": LAMBDA,
            "edge_threshold": EDGE_THRESHOLD,
            "max_iterations": MAX_ITERATIONS,
            "seeds": list(SEEDS),
        },
        "dataset": metadata,
        "disparity_labels": disparities,
        "pairwise_variance": pairwise_variance,
        "gbp_projection": "local Laplace approximation at dominant photometric mode",
        "bp_runs": bp_runs,
        "gbp_runs": gbp_runs,
        "bp_mse_mean": float(np.mean(bp_mses)),
        "bp_mse_std": float(np.std(bp_mses, ddof=1)),
        "gbp_mse_mean": float(np.mean(gbp_mses)),
        "gbp_mse_std": float(np.std(gbp_mses, ddof=1)),
        "absolute_mse_gap": mse_gap,
        "relative_mse_gap": relative_gap,
        "regions": regions,
        "mse_aligned": bool(mse_aligned),
        "spatial_kl_aligned": bool(spatial_aligned),
        "aligned": bool(mse_aligned and spatial_aligned),
        "elapsed_seconds": time.perf_counter() - started,
        "substitution": "official 375x450 Cones images bilinearly resized to the paper's 150x200 grid; disparities scaled horizontally",
    }
    print(f"  BP MSE mean±sd={result['bp_mse_mean']:.6f}±{result['bp_mse_std']:.6f}")
    print(f"  GBP MSE mean±sd={result['gbp_mse_mean']:.6f}±{result['gbp_mse_std']:.6f}")
    print(f"  BP/GBP absolute MSE gap={mse_gap:.6f}; relative gap={relative_gap:.4%}")
    print(f"  KL regions={json.dumps(regions, sort_keys=True)}")
    print(f"  Claim 6 MSE={'ALIGNED' if mse_aligned else 'DIVERGENT'}; spatial KL={'ALIGNED' if spatial_aligned else 'DIVERGENT'}")
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, "stereo_cones.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("  wrote outputs/stereo_cones.json")
    return result
