import numpy


def reproject_points(source: numpy.ndarray, target: numpy.ndarray):
    height_s, width_s = source.shape[:2]
    height_t, width_t = target.shape[:2]

    height_ratio = (height_t - 1) / (height_s - 1)
    width_ratio = (width_t - 1) / (width_s - 1)

    rows_s, cols_s = numpy.nonzero(source)
    rows_t = numpy.rint(rows_s * height_ratio).astype(numpy.int32) 
    cols_t = numpy.rint(cols_s * width_ratio).astype(numpy.int32)

    target[rows_t, cols_t] = source[rows_s, cols_s]