from src.config.config import config
from src.utils.model_utils import get_model, train_batch, validate_batch
from src.config.config import config
from tqdm import tqdm


class CustomFasterRCNN:
    """
    Custom implementation of Faster R-CNN for object detection.

    This class encapsulates the Faster R-CNN model for object detection and provides training functionality.

    Attributes:
    -----------
    model (torch.nn.Module): The Faster R-CNN model.
    device: The device (CPU or GPU) on which the model is loaded.
    """
    def __init__(self):
        """
        Initialize the CustomFasterRCNN object.
        """
        self.model = get_model(len(config['LABELS']) + 1).to(config['DEVICE'])

    def train(self, train_dataloader, optim, n_epochs, valid_dataloader=None):
        """
        Train the Faster R-CNN model.

        This method performs the training of the Faster R-CNN model using the specified data loaders.

        Parameters:
        -----------
        train_dataloader: The data loader for the training dataset.
        optimizer: The optimizer used for training.
        n_epochs (int): The number of training epochs.
        valid_dataloader (optional): The data loader for the validation dataset. Default is None.

        Returns:
        --------
        dict: A dictionary containing training and validation loss history.
        """
        history = {
            'train': {
                'trn_loss': [],
                'trn_loc_loss': [],
                'trn_regr_loss': [],
                'trn_loss_objectness': [],
                'trn_loss_rpn_box_reg': []
            },
            'valid': {
                'val_loss': [],
                'val_loc_loss': [],
                'val_regr_loss': [],
                'val_loss_objectness': [],
                'val_loss_rpn_box_reg': []
            }
        }

        for epoch in range(n_epochs):

            # Training phase
            tqdm_train_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}, Training")
            for i, batch in enumerate(tqdm_train_dataloader):
                loss, losses = train_batch(batch, self.model, optim)
                loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in
                                                                          ['loss_classifier', 'loss_box_reg',
                                                                           'loss_objectness', 'loss_rpn_box_reg']]
                history['train']['trn_loss'].append(loss.item())
                history['train']['trn_loc_loss'].append(loc_loss.item())
                history['train']['trn_regr_loss'].append(regr_loss.item())
                history['train']['trn_loss_objectness'].append(loss_objectness.item())
                history['train']['trn_loss_rpn_box_reg'].append(loss_rpn_box_reg.item())

                # Log loss on tqdm progress bar
                tqdm_train_dataloader.set_postfix(loss=loss.item(), loc_loss=loc_loss.item(),
                                                  regr_loss=regr_loss.item(), obj_loss=loss_objectness.item(),
                                                  rpn_loss=loss_rpn_box_reg.item())

            # Validation phase (if provided)
            if valid_dataloader is not None:
                tqdm_valid_dataloader = tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}, Validation")
                for i, batch in enumerate(tqdm_valid_dataloader):
                    loss, losses = validate_batch(batch, self.model)
                    loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = [losses[k] for k in
                                                                              ['loss_classifier', 'loss_box_reg',
                                                                               'loss_objectness', 'loss_rpn_box_reg']]
                    history['valid']['val_loss'].append(loss.item())
                    history['valid']['val_loc_loss'].append(loc_loss.item())
                    history['valid']['val_regr_loss'].append(regr_loss.item())
                    history['valid']['val_loss_objectness'].append(loss_objectness.item())
                    history['valid']['val_loss_rpn_box_reg'].append(loss_rpn_box_reg.item())

                    # Log loss on tqdm progress bar
                    tqdm_valid_dataloader.set_postfix(loss=loss.item(), loc_loss=loc_loss.item(),
                                                      regr_loss=regr_loss.item(), obj_loss=loss_objectness.item(),
                                                      rpn_loss=loss_rpn_box_reg.item())

        return history

    def parameters(self):
        """

        Returns:
        --------
        dict: A dictionary containing training and validation loss history.
        """
        return self.model.parameters()
