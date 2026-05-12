import time
import numpy as np
import torch
import argparse
from optimizer import build_and_save_model
from dataloaders import load_dataloader
from transforms import get_transforms
from optimizer import get_optimizer_and_scheduler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_conf_matrix(conf_matrix, num_classes, pred_batch, mask_batch):
    """ Calculate confusion matrix and add to passed confusion matrix.
    
    Args:
        conf_matrix (np array): confusion matrix size 
            (num_classes + 1, num_classes + 1)
        num_classes (int): number of classes being segmented
        pred_batch (np array): NxM or BxNxM
        mask_batch (np array): NxM or BxNxM
    """
    
    N = num_classes + 1

    if len(pred_batch.shape) == 2:
        pred_batch = np.expand_dims(pred_batch, 0)
        mask_batch = np.expand_dims(mask_batch, 0)
    for pred, mask in zip(pred_batch, mask_batch):
        conf_matrix += np.bincount(
            N * pred.reshape(-1) + mask.reshape(-1), minlength=N ** 2
        ).reshape(N, N)
    return conf_matrix

def evaluate(num_classes, conf_matrix):
    """
    Compute per-class accuracy and IoU from a confusion matrix.

    The confusion matrix is expected to have an extra row/column for an
    ignore/background class (shape: (num_classes + 1, num_classes + 1)).
    Only the first `num_classes` rows and columns are evaluated; the final
    row/column is treated as a void/ignore label and excluded.

    Parameters:
        num_classes (int): Number of foreground classes to evaluate.
        conf_matrix (np.ndarray): Square confusion matrix of shape
            (num_classes + 1, num_classes + 1), where entry [i, j] is the
            number of pixels of true class i predicted as class j.

    Returns:
        iou (np.ndarray): Per-class Intersection over Union of shape
            (num_classes,). Classes with no ground-truth pixels are NaN.
        acc (np.ndarray): Per-class accuracy (recall) of shape
            (num_classes,). Classes with no ground-truth pixels are NaN.
    """

    acc = np.full(num_classes, np.nan, dtype=np.float64)
    iou = np.full(num_classes, np.nan, dtype=np.float64)
    tp = conf_matrix.diagonal()[:-1].astype(np.float64)
    pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float64)
    pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float64)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    return iou, acc

    
def acc_metric(predb, yb):
    """
    Compute batch accuracy for a multi-class classifier.

    Parameters:
        predb (torch.Tensor): Raw logits or probabilities of shape
            (N, C), where N is batch size and C is number of classes.
        yb (torch.Tensor): Ground-truth class indices of shape (N,).

    Returns:
        torch.Tensor: Scalar mean accuracy over the batch, in [0.0, 1.0].
    """
    return (predb.argmax(dim=1) == yb).float().mean()

def train(model, num_classes, train_dl, val_dl, loss_fn, optimizer,
          epochs=91, lr_scheduler=None, save_best_val=False,
          model_file='model.tar', accumulation_steps=4, val_epoch=2):
    """
    Train a segmentation model with gradient accumulation and periodic validation.

    Parameters:
        model (torch.nn.Module): Segmentation model to train.
        num_classes (int): Number of foreground classes.
        train_dl (DataLoader): Training dataloader. Batches must be dicts
            with keys 'image' and 'mask'.
        val_dl (DataLoader): Validation dataloader. Same format as train_dl.
        loss_fn (callable): Loss function with signature
            loss_fn(outputs, masks) -> scalar tensor.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        epochs (int): Total number of training epochs. Default: 91.
        lr_scheduler (optional): Scheduler with a .step() method called
            after each optimizer step. Default: None.
        save_best_val (bool): If True, saves model state dict to
            model_file whenever validation loss improves. Default: False.
        model_file (str): Path for the checkpoint file. Default: 'model.tar'.
        accumulation_steps (int): Number of batches to accumulate gradients
            over before calling optimizer.step(). Default: 4.
        val_epoch (int): Validation runs on epochs where
            epoch % val_epoch == 0. Default: 2.

    Returns:
        train_loss (list[float]): Mean training loss recorded each epoch.
        valid_loss (list[float]): Mean validation loss recorded on each
            validated epoch.
    """
    start = time.time()
    train_loss, valid_loss = [], []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    max_batchnum = (len(train_dl) // accumulation_steps) * accumulation_steps
    logging.info("Using {} batches ({} images)".format(
        max_batchnum, max_batchnum * train_dl.batch_size))

    top_val_loss = float('inf')

    for epoch in range(epochs):
        logging.info('Epoch {}/{}'.format(epoch, epochs - 1))
        optimizer.zero_grad()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
                dataloader = train_dl
            else:
                if epoch % val_epoch != 0:
                    continue
                model.train(False)
                dataloader = val_dl
                conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

            running_loss = 0.0

            for batch_ind, batch in enumerate(dataloader):
                if batch_ind >= max_batchnum:
                    break

                im_batch = batch['image'].to(device)
                masks = batch['mask'][:, 24:-24, 24:-24].to(device)

                if phase == 'train':
                    outputs = model(im_batch)
                    loss = loss_fn(outputs, masks)
                    loss.backward()
                    if (batch_ind + 1) % accumulation_steps == 0:
                        optimizer.step()
                        if lr_scheduler:
                            lr_scheduler.step()
                        model.zero_grad()
                else:
                    with torch.no_grad():
                        outputs = model(im_batch)
                        loss = loss_fn(outputs, masks)
                        np_preds = np.argmax(outputs.cpu().numpy(), axis=1)
                        np_masks = masks.cpu().numpy()
                        conf_matrix = get_conf_matrix(conf_matrix, num_classes,
                                                      np_preds, np_masks)

                running_loss += loss.item() * dataloader.batch_size

            epoch_loss = running_loss / len(dataloader.dataset)
            logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val':
                valid_loss.append(epoch_loss)
                if epoch_loss < top_val_loss:
                    top_val_loss = epoch_loss
                    if save_best_val:
                        torch.save(model.state_dict(), model_file)
                        logging.info('Saved new best model. Val loss: {:.4f}'.format(epoch_loss))
                iou, acc = evaluate(num_classes, conf_matrix)
                logging.info('IOU: {}, Acc: {}'.format(iou, acc))
            else:
                train_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss

def main():
    parser = argparse.ArgumentParser(description='Train UNETTraditional on thermal segmentation dataset')
    parser.add_argument('--clips_dir', type=str, help='path to clips directory')
    parser.add_argument('--save_model', default=True, help='whether to save the trained model')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for optimizer')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='number of batches to accumulate gradients over')
    parser.add_argument('--epochs', type=int, default=91, help='number of epochs to train for')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer (not applicable for UNETTraditional)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training (effective batch size is batch_size * accumulation_steps)')
    parser.add_argument('--num_workers', type=int, default=7, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='whether to pin memory in dataloader')
    args = parser.parse_args()

    total_train_loss = []
    total_val_loss = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = torch.nn.NLLLoss(weight=torch.tensor([.01, .99]).to(device))
    model, model_file = build_and_save_model(args.lr, args.accumulation_steps, args.epochs)
    optimizer_none, scheduler_none, optimizer_easy, scheduler_easy, optimizer_hard, scheduler_hard = get_optimizer_and_scheduler(model, args.lr, args.momentum)

    no_aug_epochs = 1
    easy_epochs = 30
    hard_epochs = 60

    none_transforms, easy_transforms, hard_transforms = get_transforms(args.clips_dir)

    train_dataloader_no_aug   = load_dataloader(args.clips_dir, none_transforms, split='train', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    train_dataloader_easy_aug = load_dataloader(args.clips_dir, easy_transforms, split='train', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    train_dataloader_hard_aug = load_dataloader(args.clips_dir, hard_transforms, split='train', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_dataloader = load_dataloader(args.clips_dir, hard_transforms, split='val', batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)

    train_loss, val_loss = train(model, 2, train_dataloader_no_aug, val_dataloader,
                                 loss_fn, optimizer_none, epochs=no_aug_epochs,
                                 lr_scheduler=scheduler_none, save_best_val=True,
                                 model_file=model_file, val_epoch=1)
    total_train_loss.extend(train_loss)
    total_val_loss.extend(val_loss)

    train_loss, val_loss = train(model, 2, train_dataloader_easy_aug, val_dataloader,
                                 loss_fn, optimizer_easy, epochs=easy_epochs,
                                 lr_scheduler=scheduler_easy, save_best_val=True,
                                 model_file=model_file)
    total_train_loss.extend(train_loss)
    total_val_loss.extend(val_loss)

    train_loss, val_loss = train(model, 2, train_dataloader_hard_aug, val_dataloader,
                                 loss_fn, optimizer_hard, epochs=hard_epochs,
                                 lr_scheduler=scheduler_hard, save_best_val=True,
                                 model_file=model_file)
    total_train_loss.extend(train_loss)
    total_val_loss.extend(val_loss)

    if args.save_model:
        torch.save(model.state_dict(), model_file)

if __name__ == "__main__":
    main()
