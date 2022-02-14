import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data.cifar import LightningCifar
from hypernetworks.hypernetworks import HyperNetwork
from hypernetworks.target_models import BatchTargetModel, TargetModel
from models.vision import ConvTaskEnsembleCIFAR, ResNet, ResNet9, SmallResNet
from train.trainer import LightningClassifierTask


def get_resnet9(in_channels, num_classes, n):
    model = ResNet9(in_channels=in_channels, num_classes=num_classes)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--dev', type=int, default=0, metavar='d',
                        help='fast dev run (default: 0)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='l',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--bs', type=int, default=50, metavar='b',
                        help='batch size (default: 50)')
    
    parser.add_argument('--nonlin', type=str, default="linear", metavar='n',
                        help='non linearities (default: linear), other: MLPnobias, MLPwithbias, affine')

    parser.add_argument('--input', type=str, default="task", metavar='i',
                        help='input (default: "task")')
    parser.add_argument('--model', type=str, default="hnet", metavar='m',
                        help='model (default: "hnet") other: "experts"')
    args = parser.parse_args()
    max_epochs = 100
    batch_size = args.bs # 50
    use_sgd = False
    learning_rate = args.lr
    patience = 10
    monitor = "Val Loss"
    log_dir = "split_cifar"
    csv_log_dir = "split_cifar_csv"
    ckpt_path = "split_cifar_ckpt"
    accumulate_grad_batches = 3
    gradient_clip_val = 1.0
    fast_dev_run = args.dev != 0
    lr_reduce = True

    in_channels = 3
    hnet = "sparse"
    input_type = args.input
    latent_size = 256
    n = 1
    base = 2
    distribution = "normal"
    connectivity_type = "linear-decrease"
    connectivity = 3 # not used for linear decrease
    if args.nonlin == "linear":
        activation = "none"
        bias = False
    elif args.nonlin == "affine":
        bias = True
        raise NotImplementedError()
    elif args.nonlin == "MLPbias":
        bias = True
        raise NotImplementedError()
    elif args.nonlin == "MLPnobias":
        bias = False
        raise NotImplementedError()
    else:
        raise ValueError()
    batch = (input_type == "input" or input_type == "input-task")
    sigma = torch.Tensor([latent_size//4])
    step = 1

    model_to_test = args.model

    print(f"learning_rate: {learning_rate}")
    print(f"input_type: {input_type}")
    print(f"batch_size: {batch_size}")
    print(f"model to test: {model_to_test}")
    print(f"non linearity: {args.nonlin}")
    print(f"fast_dev_run: {fast_dev_run}")

    resnet = ResNet
    resnet_name = resnet.__name__

    num_class_per_task = 10
    num_classes = num_class_per_task
    n_classes = 100
    num_tasks = n_classes // num_class_per_task

    data = LightningCifar(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=num_tasks)

    for version in range(5):

        if model_to_test == "hnet":
            name = "-".join([hnet, input_type, activation, f"bias={bias}", distribution,
                            connectivity_type, resnet_name, f"stp={step}"])
        else:
            name = "-".join(["Experts", resnet_name])


        target_model = resnet(
            in_channels=in_channels, num_classes=num_classes, n=n)
        if batch:
            encoder = resnet(in_channels=in_channels,
                             num_classes=latent_size, n=1)
        else:
            encoder = None

        if batch:
            batch_target_model = BatchTargetModel(
                batch_size=batch_size, target_model=target_model)
        else:
            batch_target_model = TargetModel(target_model=target_model)

        if model_to_test == "hnet":
            model = HyperNetwork(batch_target_model=batch_target_model, hnet=hnet, input_type=input_type, encoder=encoder, latent_size=latent_size, batch=batch, sigma=sigma,
                                 base=base, num_tasks=num_tasks, distribution=distribution, connectivity_type=connectivity_type, connectivity=connectivity, activation=activation, step=step)
        elif model_to_test == "experts":
            model = ConvTaskEnsembleCIFAR(
                resnet, nbr_task=num_tasks, in_channels=in_channels, n=1, num_classes=num_classes)
        else:
            raise ValueError()

        logger = TensorBoardLogger(save_dir=log_dir, name=name)
        csv_logger = CSVLogger(save_dir=csv_log_dir, name=name)

        # Callbacks
        early_stopping_callback = EarlyStopping(
            monitor=monitor, patience=patience)
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=ckpt_path,
            filename=name + "-cifar-{epoch:02d}",
        )
        lr_monitor_callback = LearningRateMonitor()

        pl_model = LightningClassifierTask(model=model, batch_size=batch_size, patience=patience, monitor=monitor,
                                           latent_size=latent_size, learning_rate=learning_rate, use_sgd=use_sgd, lr_reduce=lr_reduce)

        trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=False, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])  # reload_dataloaders_every_n_epochs=1

        trainer.fit(pl_model, data)
        if not fast_dev_run:
            trainer.test(ckpt_path="best", dataloaders=data)
