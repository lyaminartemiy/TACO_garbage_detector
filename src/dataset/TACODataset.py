class TACODataset(torch.utils.data.Dataset):
    """
        Class for TACO Dataset.
    """

    def __init__(self, coco_object: COCO, imgs: list, categories: list, anns: list,
                 dataset_path: str, transform=f'T.Resize((224, 224))'):
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

        bboxes_array, categories = parse_bboxes_and_categories(id_image, self.anns)
        resized_image, resized_bboxes = transform_image_with_bbox(image, bboxes_array, DESIRED_SIZE,
                                                                  self.transform, (width, height))
        labels =  [get_id_of_category(cat) for cat in categories]
        categories = torch.tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([id_image])
        area = (resized_bboxes[:, 3] - resized_bboxes[:, 1]) * (resized_bboxes[:, 2] - resized_bboxes[:, 0])
        iscrowd = torch.zeros((len(bboxes_array),), dtype=torch.int64)

        target = {}
        target["boxes"] = resized_bboxes
        target["labels"] = categories
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        return resized_image, target

    def __len__(self):
        return len(imgs)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

