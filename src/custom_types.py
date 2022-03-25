from collections import namedtuple
import numpy as np
from typing import NewType, List

Image = NewType("Image", np.ndarray)
Label = NewType("Label", str)
TransformedImage = NewType("TransformedImage", List[Image])
TransformedLabel = NewType("TransformedLabel", np.ndarray)
ImageLabelPair = namedtuple("ImageLabelPair", "image label")
XDataSet = NewType("XDataSet", np.ndarray)
YDataSet = NewType("YDataSet", np.ndarray)
