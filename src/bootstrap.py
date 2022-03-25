from custom_types import DataSet
from data_retrieval import get_data, get_pre_encoded_dataset
from processing import get_processed_x_y, get_x_y_preprocessors


def get_encoded_x_y(
    image_folder_path: str,
    data_path: str,
    scale_height: bool,
    desired_width: int,
    desired_height: int,
) -> DataSet:
    data = get_data(data_path=data_path)
    images, labels = get_pre_encoded_dataset(data, image_folder_path=image_folder_path)
    x_encoder, y_encoder = get_x_y_preprocessors(
        images, labels, desired_height, desired_width, scale_height
    )
    x, y = get_processed_x_y(images, labels, x_encoder, y_encoder)

    return x, y
