import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max


def detect_tx_coords(
    matrix: np.ndarray,
    percentile: float = 98.0,
    min_distance: int = 10,
    exclude_border: bool = False,
    threshold_abs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    从二维信号矩阵中检测Transmitter峰值coordinate。

    Args:
        matrix: 2D signal-strength matrix。
        percentile: percentile used to compute the adaptive threshold when threshold_abs is None。
        min_distance: minimum pixel distance between peaks。
        exclude_border: whether to exclude peaks near image borders。
        threshold_abs: absolute threshold；if None, compute automatically from percentile。

    Returns:
        coords: shape (N, 2) 的coordinate数组，format (y, x)。
        powers: 每个coordinate对应的峰值强度，shape (N,)。
        used_threshold: threshold actually used for this detection。
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be a 2D array，current number of dimensions: {matrix.ndim}")

    used_threshold = (
        float(np.percentile(matrix, percentile))
        if threshold_abs is None
        else float(threshold_abs)
    )

    coords = peak_local_max(
        matrix,
        min_distance=min_distance,
        threshold_abs=used_threshold,
        exclude_border=exclude_border,
    )

    if len(coords) == 0:
        return coords, np.array([], dtype=matrix.dtype), used_threshold

    powers = matrix[coords[:, 0], coords[:, 1]]
    return coords, powers, used_threshold


def detect_tx_from_png(
    png_path: str,
    percentile: float = 98.0,
    min_distance: int = 10,
    exclude_border: bool = False,
    threshold_abs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    从 PNG 文件读取矩阵并检测Transmittercoordinate。

    Returns:
        coords, powers, used_threshold, matrix
    """
    matrix = plt.imread(png_path)
    # Support RGBA/RGB images by using the first channel as the intensity matrix.
    if matrix.ndim == 3:
        matrix = matrix[..., 0]

    coords, powers, used_threshold = detect_tx_coords(
        matrix=matrix,
        percentile=percentile,
        min_distance=min_distance,
        exclude_border=exclude_border,
        threshold_abs=threshold_abs,
    )
    return coords, powers, used_threshold, matrix


__all__ = ["detect_tx_coords", "detect_tx_from_png"]
