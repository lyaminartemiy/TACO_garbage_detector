from PIL import Image
import torch
import torchvision.transforms as T


def transform_image_with_bbox(image: Image, bboxes: list, desired_size: tuple, transform: T, size: tuple):
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
    bboxes = []
    categories = []

    for ann in anns:
        if ann['image_id'] == id_image:
            bboxes.append(ann['bbox'])
            categories.append(ann['category_id'])

    return bboxes, categories


def get_id_of_category(id_name: int):
    for index, label in enumerate(LABELS):
        if id_name in LABELS[label]:
            return index + 1
    return -1