import torch
from bat_seg_models import UNETTraditional
import os

def lr_scheduler(epoch):
    """ To pass to torch optim lr_scheduler.
    
    Args:
        epoch: current epoch number
        
    return number to multiply base lr
    """
    
    if epoch < 15:
        return 1.0
    elif epoch < 30:
        return 0.2
    else:
        return 0.05

def build_and_save_model(lr=0.01, accumulation_steps=4, epochs=91, momentum=0.9, batch_size=4):
    """
    Instantiate a UNETTraditional model and construct its checkpoint file path.

    Builds a single-channel-input, two-class-output U-Net (no padding) and
    generates a deterministic filename encoding the key hyperparameters so
    that checkpoint files are self-describing and don't collide across runs.

    Args:
        lr (float): Learning rate, embedded in the filename. Default: 0.01.
        accumulation_steps (int): Gradient accumulation steps. Combined with
            `batch_size` to record the effective batch size in the filename.
            Default: 4.
        epochs (int): Total training epochs, embedded in the filename.
            Default: 91.
        momentum (float): SGD momentum, embedded in the filename. Default: 0.9.
        batch_size (int): Per-step batch size. Multiplied by
            `accumulation_steps` to compute effective batch size. Default: 4.

    Returns:
        model (UNETTraditional): Untrained model with in_channels=1,
            out_channels=2, should_pad=False.
        model_file (str): Path of the form
            './models/model_UNETTraditional_epochs_{e}_batcheff_{b}_lr_{lr}
            _momentum_{m}_aug_thermal-radiometric-heatseek.tar'.
            The ./models/ directory is created if it does not exist.

    Note:
        This function only constructs and names the model — it does not
        train it or write any file to disk. Pass model_file to train() with
        save_best_val=True to actually checkpoint the weights.
    """

    model = UNETTraditional(in_channels=1, out_channels=2, should_pad=False)
    # refer to bat_seg_models.py to set any other models; Koger uses UNETTraditional for all experiments, so we hardcode that here for simplicity
    model_name = "UNETTraditional"
    aug_type = "thermal-radiometric-heatseek"

    folder = './models'
    os.makedirs(folder, exist_ok=True)

    model_name = 'model_{}_epochs_{}_batcheff_{}_lr_{}_momentum_{}_aug_{}'.format(
        model_name, epochs, accumulation_steps*batch_size, lr, momentum, aug_type)
    
    model_file = os.path.join(folder, model_name + '.tar')
    return model, model_file

def get_optimizer_and_scheduler(model, lr=0.01, momentum=0.9):
    """
    Create three independent SGD optimizer/scheduler pairs for none, easy,
    and hard augmentation training regimes.

    Each pair is fully independent — separate parameter groups, separate
    state — so they can be used to train the same model sequentially under
    different augmentation conditions without state leaking between regimes.
    All three share the same `lr`, `momentum`, and `lr_scheduler` lambda.

    Args:
        model (torch.nn.Module): Model whose parameters all three optimizers
            will reference.
        lr (float): Initial learning rate for all three SGD instances.
            Default: 0.01.
        momentum (float): Momentum for all three SGD instances. Default: 0.9.

    Returns:
        optimizer_none (torch.optim.SGD): Optimizer for the no-augmentation regime.
        scheduler_none (LambdaLR): Scheduler paired with optimizer_none.
        optimizer_easy (torch.optim.SGD): Optimizer for the easy-augmentation regime.
        scheduler_easy (LambdaLR): Scheduler paired with optimizer_easy.
        optimizer_hard (torch.optim.SGD): Optimizer for the hard-augmentation regime.
        scheduler_hard (LambdaLR): Scheduler paired with optimizer_hard.

    Note:
        All three optimizers reference the same model.parameters() iterator
        at construction time. If the model is replaced or re-initialized
        between regimes, these optimizers will hold stale parameter references
        and must be reconstructed.
    """
    optimizer_none = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler_none = torch.optim.lr_scheduler.LambdaLR(optimizer_none, lr_scheduler, last_epoch=-1)

    optimizer_easy = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler_easy = torch.optim.lr_scheduler.LambdaLR(optimizer_easy, lr_scheduler, last_epoch=-1)

    optimizer_hard = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler_hard = torch.optim.lr_scheduler.LambdaLR(optimizer_hard, lr_scheduler, last_epoch=-1)

    return optimizer_none, scheduler_none, optimizer_easy, scheduler_easy, optimizer_hard, scheduler_hard
