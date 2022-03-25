from functools import partial
from typing import Callable
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from pipetools import pipe, foreach
import cv2
from custom_types import (
    Encoders,
    PreprocessedImage,
    PreprocessedLabel,
    Image,
    ProcessorFactory,
    XDataSet,
    YDataSet,
)


def get_scaled_dimensions(dim1: int, dim2: int, target_dim: int, reverse=False):
    aspect_ratio = dim1 / dim2
    scale_dim_by = dim2 / target_dim
    scaled_dim2 = round(dim2 / scale_dim_by)
    scaled_dim1 = round(dim1 / scale_dim_by * aspect_ratio)
    scaled_dims = (scaled_dim1, scaled_dim2)
    return scaled_dims[::-1] if reverse else scaled_dims


def change_image_to_greyscale(img: Image) -> Image:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_image(img: Image) -> Image:
    return cv2.normalize(
        img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )


def resize_image(
    img: Image, desired_width: int, desired_height: int, scale_height: bool = False
) -> Image:
    height, width, _ = img.shape
    scaled_dims = (
        get_scaled_dimensions(width, height, target_dim=desired_width)
        if scale_height
        else get_scaled_dimensions(
            height, width, target_dim=desired_height, reverse=True
        )
    )
    return cv2.resize(img, scaled_dims)


def pad_image(
    img: Image, desired_width: int, desired_height: int, scale_height: bool = False
) -> Image:
    height, width = img.shape
    bottom = 0
    top = 0
    left = 0
    right = 0

    if scale_height:
        width_padding_needed = max(desired_width - width, 0)
        width_padding_per_side = width_padding_needed // 2

        left = width_padding_per_side
        right = width_padding_per_side

        if width_padding_needed % 2 == 1:
            right += 1
    else:
        height_padding_needed = max(desired_height - height, 0)
        height_padding_per_side = height_padding_needed // 2

        top = height_padding_per_side
        bottom = height_padding_per_side

        if height_padding_needed % 2 == 1:
            top += 1

    padded_image = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT
    )

    return padded_image


def crop_image(
    img: Image, desired_height: int, desired_width: int, scale_height: bool = False
):
    height, width = img.shape

    h_low = 0
    h_high = desired_height
    w_low = 0
    w_high = desired_width

    if scale_height:
        difference = desired_width - width
        crops = difference // 2
        w_low = 0 - crops
        w_high = width + crops

        if difference % 2 == 1:
            w_high += 1
    else:
        difference = desired_height - height
        crops = difference // 2
        h_low = 0 - crops
        h_high = height + crops

        if difference % 2 == 1:
            h_high += 1

    return img[h_low:h_high, w_low:w_high]


def get_x_encoder(desired_height: int, desired_width: int, scale_height: bool = False):
    return (
        pipe
        | partial(
            resize_image,
            desired_height=desired_height,
            desired_width=desired_width,
            scale_height=scale_height,
        )
        | change_image_to_greyscale
        | normalize_image
        | partial(
            pad_image,
            desired_height=desired_height,
            desired_width=desired_width,
            scale_height=scale_height,
        )
        | partial(
            crop_image,
            desired_height=desired_height,
            desired_width=desired_width,
            scale_height=scale_height,
        )
    )


def get_x_y_preprocessors(
    x: PreprocessedImage,
    y: PreprocessedLabel,
    desired_height: int,
    desired_width: int,
    scale_height: bool = False,
) -> Encoders:
    y_encoder = OneHotEncoder(sparse=False).fit(y)
    x_encoder = get_x_encoder(desired_height, desired_width, scale_height)

    return x_encoder, y_encoder


def get_processed_x_y(
    x: PreprocessedImage, y: PreprocessedLabel, processor_factory: ProcessorFactory
):
    x_encoder, y_encoder = processor_factory(x, y)
    x_encoded = x > foreach(x_encoder) | list | np.array
    y_encoded = y_encoder.transform(y)

    return XDataSet(x_encoded), YDataSet(y_encoded)
