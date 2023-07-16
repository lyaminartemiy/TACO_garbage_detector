from src.config.config import config
from src.utils.model_utils import get_model


class CustomFasterRCNN:
    """

    """
    def __init__(self, num_classes, device):
        self.model = get_model(len(config['LABELS']) + 1).to(device)
        self.num_classes = num_classes
        self.device = device
