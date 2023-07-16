import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.config.config import config


def get_model(num_classes):
    """

    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_batch(batch, model, optim):
    """

    """
    model.train()

    imgs, targets = batch
    imgs = list(img.to(config['DEVICE']) for img in imgs)
    targets = [{k: v.to(config['DEVICE']) for k, v in t.items()} for t in targets]

    optim.zero_grad()

    losses = model(imgs, targets)
    loss = sum(loss for loss in losses.values())

    loss.backward()
    optim.step()

    return loss, losses


@torch.no_grad()
def validate_batch(batch, model):
    """

    """
    model.eval()

    imgs, targets = batch
    imgs = list(img.to(config['DEVICE']) for img in imgs)
    targets = [{k: v.to(config['DEVICE']) for k, v in t.items()} for t in targets]

    losses = model(imgs, targets)
    loss = sum(loss for loss in losses.values())

    return loss, losses
