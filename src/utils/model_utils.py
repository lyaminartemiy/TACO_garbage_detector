import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.config.config import config


def get_model(num_classes):
    """
    Get a custom Faster R-CNN model.

    This function returns a custom Faster R-CNN model based on the ResNet-50 FPN architecture,
    with the final classifier replaced to support the specified number of classes.

    Parameters:
    -----------
    num_classes (int): The number of classes (including background) for the model's classifier.

    Returns:
    --------
    torch.nn.Module: A custom Faster R-CNN model with the specified number of classes.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_batch(batch, model, optim):
    """
    Perform a single training iteration on a batch.

    This function performs a forward pass on the model and computes the loss for the given batch of data.
    It then backpropagates the gradients and updates the model's parameters using the specified optimizer.

    Parameters:
    -----------
    batch (tuple): A tuple containing the input images and target dictionaries for the batch.
    model (torch.nn.Module): The Faster R-CNN model to be trained.
    optim (torch.optim.Optimizer): The optimizer to be used for training.

    Returns:
    --------
    tuple: A tuple containing the total loss and a dictionary of individual loss components.
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
def validate_batch(batch, model, optim):
    """
    Perform a single validation iteration on a batch.

    This function performs a forward pass on the model and computes the loss for the given batch of data.
    However, it does not backpropagate gradients or update the model's parameters since it is used for validation.

    Parameters:
    -----------
    batch (tuple): A tuple containing the input images and target dictionaries for the batch.
    model (torch.nn.Module): The Faster R-CNN model to be validated.

    Returns:
    --------
    tuple: A tuple containing the total loss and a dictionary of individual loss components.
    """
    model.train()

    imgs, targets = batch
    imgs = list(img.to(config['DEVICE']) for img in imgs)
    targets = [{k: v.to(config['DEVICE']) for k, v in t.items()} for t in targets]

    optim.zero_grad()

    losses = model(imgs, targets)
    loss = sum(loss for loss in losses.values())

    return loss, losses
