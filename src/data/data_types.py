from collections import namedtuple
from typing import Callable, List, NewType, Tuple

import numpy as np
from pipetools.main import Pipe

Image = NewType("Image", np.ndarray)
Label = NewType("Label", str)

PreEncodedImages = NewType("PreEncodedImages", List[Image])
PreEncodedLabels = NewType("PreEncodedLabels", np.ndarray)

PreEncodedDataSet = namedtuple("PreEncodedDataSet", ["image", "label"])

XDataSet = NewType("XDataSet", np.ndarray)
YDataSet = NewType("YDataSet", np.ndarray)
DataSet = namedtuple("DataSet", ["x", "y"])

FeaturesEncoder = NewType("FeaturesEncoder", Pipe)
TargetEncoder = Callable[[Label], YDataSet]
Encoders = Tuple[FeaturesEncoder, TargetEncoder]
