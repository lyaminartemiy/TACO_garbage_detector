from PIL import Image
import torch
import torchvision.transforms as T
from src.config.config import config


def transform_image_with_bbox(image: Image, bboxes: list, desired_size: tuple, transform: T, size: tuple):
    """
    Transform the input image and bounding boxes to the desired size.

    Parameters:
    -----------
    image (PIL.Image.Image): The input image to be transformed.
    bboxes (list): A list of bounding boxes in the format [x, y, width, height].
    desired_size (tuple): The desired size for the transformed image (width, height).
    transform (torchvision.transforms.Compose): The transformation pipeline for the image.
    size (tuple): The original size of the input image (width, height).

    Returns:
    --------
    tuple: A tuple containing the transformed image as a torch.FloatTensor and the resized bounding boxes
           as a torch.Tensor of shape (num_bboxes, 4), where num_bboxes is the number of bounding boxes,
           and each bounding box is represented as [x_min, y_min, x_max, y_max].
    """
    resized_image = torch.FloatTensor(transform(image))
    resized_bboxes = []

    for bbx in bboxes:
        resized_bbox = [bbx[0] * (desired_size[0] / size[0]),
                        bbx[1] * (desired_size[1] / size[1]),
                        bbx[2] * (desired_size[0] / size[0]) + bbx[0] * (desired_size[0] / size[0]),
                        bbx[3] * (desired_size[1] / size[1]) + bbx[1] * (desired_size[1] / size[1])]
        resized_bboxes.append(resized_bbox)

    resized_bboxes = torch.as_tensor(resized_bboxes, dtype=torch.float32)

    return resized_image, resized_bboxes


def parse_bboxes_and_categories(id_image: int, anns: list):
    """
    Extract bounding boxes and categories for a specific image from the annotations.

    Parameters:
    -----------
    id_image (int): The ID of the image for which to extract bounding boxes and categories.
    anns (list): A list of annotations containing bounding box and category information.

    Returns:
    --------
    tuple: A tuple containing a list of bounding boxes and a list of category IDs
           corresponding to the input image ID.
    """
    bboxes = []
    categories = []

    for ann in anns:
        if ann['image_id'] == id_image:
            bboxes.append(ann['bbox'])
            categories.append(ann['category_id'])

    return bboxes, categories


def get_id_of_category(id_name: int) -> object:
    """
    Get the ID of a category based on its name.

    Parameters:
    -----------
    id_name (int): The ID of the category name to be searched.

    Returns:
    --------
    int: The ID of the category. If the category is not found, returns -1.
    """
    for index, label in enumerate(config['LABELS']):
        if id_name in config['LABELS'][label]:
            return index + 1
    return -1
