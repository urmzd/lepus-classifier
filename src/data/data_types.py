from collections import namedtuple
import numpy as np
from pipetools.main import Pipe
from typing import Callable, NewType, List, Tuple

from sklearn.preprocessing import OneHotEncoder

Image = NewType("Image", np.ndarray)
Label = NewType("Label", str)

PreEncodedImages = NewType("PreEncodedImages", List[Image])
PreEncodedLabels = NewType("PreEncodedLabels", np.ndarray)

PreEncodedDataSet = namedtuple("PreEncodedDataSet", ["image", "label"])

XDataSet = NewType("XDataSet", np.ndarray)
YDataSet = NewType("YDataSet", np.ndarray)
DataSet = namedtuple("DataSet", ["x", "y"])

FeaturesEncoder = NewType("FeaturesEncoder", Pipe)
TargetEncoderHandler = NewType("TargetEncoderHandler", OneHotEncoder)
TargetEncoder = Callable[[Label], YDataSet]
Encoders = Tuple[FeaturesEncoder, TargetEncoderHandler]
