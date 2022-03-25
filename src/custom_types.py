from collections import namedtuple
import numpy as np
from typing import Callable, NewType, List, Tuple

from sklearn.preprocessing import OneHotEncoder

Image = NewType("Image", np.ndarray)
Label = NewType("Label", str)

PreEncodedImages = NewType("PreEncodedImages", List[Image])
PreEncodedLabels = NewType("PreEncodedLabels", np.ndarray)

ImageLabelPair = namedtuple("ImageLabelPair", "image label")

XDataSet = NewType("XDataSet", np.ndarray)
YDataSet = NewType("YDataSet", np.ndarray)
DataSet = namedtuple("DataSet", "x y")

XEncoder = Callable[[Image], Image]
YEncoder = OneHotEncoder
Encoders = Tuple[XEncoder, YEncoder]

ProcessorFactory = Callable[[PreEncodedImages, PreEncodedLabels], Encoders]
