import matplotlib
import numpy as np
from quicklook import quicklook


def arr_to_image(arr: np.array) -> np.array:
    """modify array so that it is optimized for viewing"""

    # image array
    img = np.array(arr)

    img = quicklook.replace_nan(img)
    img = quicklook.percentile_clip(img, 2)
    img = quicklook.bytescale(img)
    img = quicklook.reshape_array(img)

    return img


def numpy_to_png(
    arr: np.array, dim="time", index="somedate", fpath="image.png"
) -> None:
    """Save numpy array to png"""

    # image from array
    img = arr_to_image(arr)

    # save to file
    matplotlib.image.imsave(fpath, img)
