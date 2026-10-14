import numpy
import random
import scipy
from typing import Optional

    
class NumpyCropTop:
    def __init__(self, height: int, width: int, return_left_top: bool = False):
        self._height = height
        self._width = width
        self._left_top = return_left_top

    def __call__(self, image: numpy.ndarray) -> numpy.ndarray:
        height, width = image.shape[0:2]

        assert height >= self._height and width >= self._width, \
            f"Image dimensions ({height}, {width}) are smaller than crop size ({self._height}, {self._width})"
        
        left = (width - self._width) // 2
        right = left + self._width
        top = height - self._height

        if self._left_top:
            return image[top:, left:right, ...], left, top

        return image[top:, left:right, ...]


class RandomPointsFromDepth:
    def __init__(self, preserve: float | int, background: float = -1.0, valid_threshold: int | float | None = None, 
                 valid_mask: numpy.ndarray | None = None, return_mask: bool = False, mode: str = "percent") -> None:
        """
        :param preserve: Expresses amount of points to be preserved from depth images. Example:
        if the mode is 'percent' then value 0.5 preserves 50% of points. If the mode is 'points'
        the value to the point count which will be preserved.
        :param mode: to be used together with 'preserve'
        :param background: Value of pixels that are not copied from provided depth map.
        :param valid_threshold: only values larger than valid threshold will be preserved
        :param return_mask: if true, the callable returns also a mask with values set to true in positions of sampled values
        """

        assert not (valid_threshold is not None and valid_mask is not None), "only one of valid_threshold or valid_mask may be provided"
        assert mode == "percent" or mode == "points"

        self._preserve = preserve
        self._mode = mode
        self._background = background
        self._valid_threshold = valid_threshold
        self._valid_mask = valid_mask
        self._return_mask = return_mask

    def __call__(self, depth: numpy.ndarray) -> numpy.ndarray:
        assert depth.ndim == 2, "depth must be only two dimensional"
        height, width = depth.shape
        d = depth.ravel()

        if self._valid_threshold is not None:
            population = list(numpy.ravel(numpy.argwhere(d > self._valid_threshold)))
        elif self._valid_mask is not None:
            population = list(numpy.ravel(numpy.argwhere(self._valid_mask.ravel())))
        else:
            population = list(range(0, d.shape[0]))

        if self._mode == "percent":
            k = min(round(d.shape[0] * self._preserve), len(population))
        else:
            # preserves given point count
            k = min(self._preserve , len(population))

        idxs = random.sample(population=population, k=k)
        result = numpy.full(shape=d.shape, fill_value=self._background, dtype=d.dtype)
        result[idxs] = d[idxs]
        result = numpy.reshape(result, (height, width))

        if self._return_mask:
            mask = numpy.full(shape=d.shape, fill_value=False, dtype=numpy.bool_)
            mask[idxs] = True
            mask = numpy.reshape(mask, (height, width))
            return result, mask

        return result


class InterpolateLidar:
    def __init__(self, interpolation: str = "linear", fill_value: float | None = None):
        self._interpolation = interpolation
        self._fill_value = fill_value

    def __call__(self, depth: numpy.ndarray, mask: Optional[numpy.ndarray] = None, valid_threshold: Optional[int | float] = None) -> numpy.ndarray:
        assert depth.ndim == 2, "depth must be two dimensional"
        assert (mask is not None) ^ (valid_threshold is not None), "only one of mask or valid_threshold must be provided"
        height, width = depth.shape

        if mask is not None:
            points = numpy.argwhere(mask)
        else:
            points = numpy.argwhere(depth > valid_threshold)
        
        values = depth[points[:, 0], points[:, 1]]
        points = numpy.flip(points, axis=1)
        X, Y = numpy.meshgrid(numpy.arange(0, width, 1), numpy.arange(0, height, 1))
        result = scipy.interpolate.griddata(points, values, (X, Y), self._interpolation, fill_value=self._fill_value)
        return result.astype(numpy.float32)
