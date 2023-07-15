import os
from torch.utils.data import Dataset
from src.utils.ann_transform import *
from src.config.config import config


def collate_fn(batch):
    return tuple(zip(*batch))


class TACODataset(Dataset):
    """
        Class for TACO Dataset.
    """

    def __init__(self, imgs: list, categories: list, anns: list, dataset_path: str, transform=f'T.Resize((224, 224))'):
        self.imgs = imgs
        self.categories = categories
        self.anns = anns
        self.dataset_path = dataset_path
        self.transform = transform

    def __getitem__(self, id_image: int):

        file_name = self.imgs[id_image]["file_name"]
        width, height = self.imgs[id_image]['width'], self.imgs[id_image]['height']

        image_path = os.path.join(self.dataset_path, file_name)
        image = Image.open(image_path).convert('RGB')

        bboxes_array, labels = parse_bboxes_and_categories(id_image, self.anns)
        resized_image, resized_bboxes = transform_image_with_bbox(image, bboxes_array, config['DESIRED_SIZE'],
                                                                  self.transform, (width, height))
        labels = [get_id_of_category(cat) for cat in labels]
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": resized_bboxes, "labels": labels}
        return resized_image, target

    def __len__(self):
        return len(self.imgs)
