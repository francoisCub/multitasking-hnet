import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data.cifar import LightningCifar
from data.cifar_tasks import LightningCifarTasks
from hypernetworks.hypernetworks import HyperNetwork
from hypernetworks.modules import get_sparsify
from hypernetworks.target_models import BatchTargetModel, TargetModel
from models.vision import ConvTaskEnsembleCIFAR, ResNet, ResNet32x32, ResNet9, SmallResNet, get_resnet18
from train.trainer import LightningClassifierTask
from hypernetworks.utils import compute_nbr_params, sparsify_resnet
from random import randint

def get_comparison_name(**kwargs):
    return "-".join([f"eid={kwargs['exp_id']}", f"d={kwargs['distribution']}", f"c={kwargs['connectivity_type']}", f"lin={kwargs['nonlin']}", f"lr={kwargs['lr']}", f"n={kwargs['normalize']}"])

def get_comparison_name2(**kwargs):
    return "-".join([kwargs['hnet'], kwargs['distribution'], f"optim={kwargs['use_optim']}", f"lin={kwargs['args.nonlin']}",
                           kwargs['connectivity_type'], f"lr={kwargs['learning_rate']:.4f}", f"lr_red={kwargs['lr_reduce']}_{kwargs['lr_reduce_factor']}", f"norm={kwargs['normalize']}", f"eid={kwargs['exp_id']}"])

def get_sparsify_name(model_to_test, **kwargs):
    if model_to_test == "hnet":
        return "-".join([kwargs['hnet'], kwargs['distribution'], f"optim={kwargs['use_optim']}", f"lin={kwargs['nonlin']}",
                           kwargs['connectivity_type'], f"lr={kwargs['learning_rate']:.4f}", f"lr_red={kwargs['lr_reduce']}_{kwargs['lr_reduce_factor']}", f"norm={kwargs['normalize']}", f"eid={kwargs['exp_id']}", f"tg_sparsity={kwargs['target_sparsity']}"])
    else:
        return "-".join(["Experts", f"optim={kwargs['use_optim']}", f"lr={kwargs['learning_rate']:.4f}", f"lr_red={kwargs['lr_reduce']}_{kwargs['lr_reduce_factor']}", f"eid={kwargs['exp_id']}", f"norm={kwargs['normalize']}", f"tg_sp={kwargs['target_sparsity']}"])

def get_target_net(name):
    networks = {"ResNet32x32": ResNet32x32,
    "ResNet18": get_resnet18}
    return networks[name]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--dev', type=int, default=0, metavar='d',
                        help='fast dev run (default: 0)')
    parser.add_argument('--lrred', type=int, default=1, metavar='r',
                        help='lr reduction (default: 1 (true))')
    parser.add_argument('--name', type=str, default="noname", metavar='n',
                        help='Experiment name (default: noname)')
    parser.add_argument('--norm', type=int, default=1, metavar='r',
                        help='normalization of input (default: 1 (true))')
    parser.add_argument('--lsize', type=int, default=512, metavar='l',
                        help='latent size d (default: 512)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='l',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--sigma', type=float, default=-1.0, metavar='s',
                        help='sigma (default: latent_size/4)')
    parser.add_argument('--target_sparsity', type=float, default=0.0, metavar='s',
                        help='target network uniform sparsity (default: 0.0, no sparsity)')
    parser.add_argument('--bs', type=int, default=50, metavar='b',
                        help='batch size (default: 50)')
    
    parser.add_argument('--nonlin', type=str, default="linear", metavar='n',
                        help='non linearities (default: linear), other: MLPnobias, MLPbias, affine')
    
    parser.add_argument('--ctype', type=str, default="linear-decrease", metavar='c',
                        help='connectivity type (default: linear-decrease), other: linear-decrease, exponential-decrease, constant')

    parser.add_argument('--input', type=str, default="task", metavar='i',
                        help='input (default: "task")')
    parser.add_argument('--data', type=str, default="cifar100", metavar='d',
                        help='dataset (default: "cifar100") other: cifar10, mnist')
    parser.add_argument('--distrib', type=str, default="normal", metavar='d',
                        help='distribution (default: "normal")')
    parser.add_argument('--model', type=str, default="hnet", metavar='m',
                        help='model (default: "hnet") other: "experts"')
    parser.add_argument('--hnet', type=str, default="sparse", metavar='h',
                        help='hnet type (default: "sparse") other: "chunked"')
    parser.add_argument('--target', type=str, default="ResNet32x32", metavar='h',
                        help='target network (default: "ResNet32x32") other: "TBA"')
    parser.add_argument('--patience', type=int, default=10, metavar='h',
                        help='patience (default: 10)')
    
    parser.add_argument('--expid', type=int, default=0, metavar='e',
                        help='experiment ID (default: 0) ')
    parser.add_argument('--cls_per_task', type=int, default=10, metavar='e',
                        help='Number of classes per task (cls_per_task: 10) ')
    parser.add_argument('--num_tasks', type=int, default=-1, metavar='e',
                        help='Number of tasks (defalut: -1: num_classes/cls_per_task) ')
    parser.add_argument('--nchunks', type=int, default=23, metavar='e',
                        help='Number of chunks (default: 23) ')
    parser.add_argument('--trials', type=int, default=1, metavar='t',
                        help='Number of trials of same method (default: 1) ')
    parser.add_argument('--optim', type=str, default="adam", metavar='o',
                        help='optimizer (default: "adam") other: "radam, sgd"')
    parser.add_argument('--special_training', type=str, default="normal", metavar='t',
                        help='special training (default: "normal") other: "fewshot, ..."')

    args = parser.parse_args()
    exp_id = args.expid
    max_epochs = 300
    batch_size = args.bs # 50
    use_sgd = False
    use_optim = args.optim
    learning_rate = args.lr # 0.001
    patience = args.patience
    monitor = "Val Acc"
    mode = "max"
    log_dir = args.name
    csv_log_dir = f"{args.name}_csv"
    ckpt_path = f"{args.name}_ckpt"
    accumulate_grad_batches = 3 # Always
    gradient_clip_val = 1.0 # Always
    fast_dev_run = args.dev != 0
    lr_reduce = args.lrred == 1 # False for first 2 experiments
    lr_reduce_factor = 0.5 # none for first 2 experimetns

    in_channels = 3 # Data
    hnet = args.hnet
    input_type = args.input
    normalize = args.norm == 1
    latent_size = args.lsize # 512 or 10000
    n = 5 # Resnet depth parameter
    base = 2 # always
    distribution = args.distrib
    connectivity_type = args.ctype #"exponential-decrease"
    connectivity = 3 # not used for linear decrease
    nbr_chunks = args.nchunks
    if args.nonlin == "linear":
        activation = "none"
        bias = False
    elif args.nonlin == "affine":
        bias = True
        activation = "none"
    elif args.nonlin == "MLPbias":
        bias = True
        activation = "prelu"
    elif args.nonlin == "MLPnobias":
        bias = False
        activation = "prelu"
    else:
        raise ValueError()
    batch = (input_type == "input" or input_type == "input-task")
    if args.sigma < 0:
        sigma = torch.Tensor([latent_size//4])
    else:
        sigma =  torch.Tensor([args.sigma])
    step = 1
    seed = randint(0, 2**31)

    target_sparsity = args.target_sparsity

    model_to_test = args.model

    special_training = args.special_training

    print(f"learning_rate: {learning_rate}")
    print(f"hnet_type: {hnet}")
    print(f"input_type: {input_type}")
    print(f"batch_size: {batch_size}")
    print(f"model to test: {model_to_test}")
    print(f"non linearity: {args.nonlin}")
    print(f"fast_dev_run: {fast_dev_run}")
    print(f"patience={patience}")

    # # ResNet18
    # resnet = get_resnet18
    # resnet_name = "ResNet18"

    # # ResNet18
    # resnet = ResNet32x32
    resnet_name = args.target
    resnet = get_target_net(resnet_name)

    num_class_per_task = args.cls_per_task
    num_classes = num_class_per_task
    
    if args.data == "cifar100":
        n_classes = 100
        num_tasks = n_classes // num_class_per_task
        data = LightningCifar(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=num_tasks)
    elif args.data == "cifar10":
        n_classes = 10
        num_tasks = n_classes // num_class_per_task
        data = LightningCifar(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=num_tasks)
    else:
        raise NotImplementedError("Only cifar100 is implemented")
    
    if args.num_tasks > 0:
        num_tasks = args.num_tasks # can discard data previously built

    for version in range(args.trials):

        name = get_sparsify_name(model_to_test, hnet=hnet, distribution=distribution, use_optim=use_optim, nonlin=args.nonlin, connectivity_type=connectivity_type, learning_rate=learning_rate, lr_reduce=lr_reduce, lr_reduce_factor=lr_reduce_factor, exp_id=exp_id, patience=patience, normalize=normalize, target_sparsity=target_sparsity)

        target_model = resnet(
            in_channels=in_channels, num_classes=num_classes, n=n)
        if batch:
            encoder = resnet(in_channels=in_channels,
                             num_classes=latent_size, n=1)
        else:
            encoder = None
        
        # Post processing
        if 0 < target_sparsity < 1:
            post_processing = get_sparsify(total_size=compute_nbr_params(target_model), sparsity=target_sparsity)
        else:
            post_processing =None

        if batch:
            batch_target_model = BatchTargetModel(
                batch_size=batch_size, target_model=target_model)
        else:
            batch_target_model = TargetModel(target_model=target_model)

        if model_to_test == "hnet":
            model = HyperNetwork(batch_target_model=batch_target_model, hnet=hnet, input_type=input_type, encoder=encoder, latent_size=latent_size, batch=batch, sigma=sigma,
                                 base=base, num_tasks=num_tasks, distribution=distribution, connectivity_type=connectivity_type, connectivity=connectivity, activation=activation, step=step, nbr_chunks=nbr_chunks, bias_sparse=bias, normalize=normalize, post_processing=post_processing, seed=seed)
        elif model_to_test == "experts":
            # TODO adapt for other dataset and Target network
            if args.data == "cifar100" or args.data == "cifar10":
                model = ConvTaskEnsembleCIFAR(
                resnet, nbr_task=num_tasks, in_channels=in_channels, n=n, num_classes=num_classes)
            else:
                raise NotImplementedError("Only cifar100")
            if target_sparsity > 0:
                model = sparsify_resnet(model, target_sparsity)
        else:
            raise ValueError()

        logger = TensorBoardLogger(save_dir=log_dir, name=name)
        csv_logger = CSVLogger(save_dir=csv_log_dir, name=name)

        # Callbacks
        early_stopping_callback = EarlyStopping(
            monitor=monitor, patience=patience, mode=mode)
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            dirpath=ckpt_path,
            filename=name + f"-{args.name}" + "-{epoch:02d}",
        )
        lr_monitor_callback = LearningRateMonitor()

        pl_model = LightningClassifierTask(model=model, batch_size=batch_size, patience=patience, monitor=monitor, mode=mode, lr_reduce_factor=lr_reduce_factor,
                                           latent_size=latent_size, learning_rate=learning_rate, use_sgd=use_sgd, lr_reduce=lr_reduce, use_optim=use_optim,
                                           batch_target_model=batch_target_model.__class__.__name__, hnet=hnet, input_type=input_type, encoder=encoder.__class__.__name__, batch=batch, sigma=sigma.item(), # next model params
                                           base=base, num_tasks=num_tasks, distribution=distribution, connectivity_type=connectivity_type, connectivity=connectivity, activation=activation, step=step, seed=seed,
                                           nbr_chunks=nbr_chunks, bias_sparse=bias, normalize=normalize, name=args.name, resnet_name=resnet_name, num_class_per_task=num_class_per_task, data=args.data, target_name=args.target, trials=args.trials, target_sparsity=target_sparsity)

        trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=True, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])  # reload_dataloaders_every_n_epochs=1

        if special_training == "normal":
            trainer.fit(pl_model, data)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data)
        elif special_training == "fewshot":
            data_1 = LightningCifar(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=num_tasks, tasks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            data_2 = LightningCifar(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=num_tasks, tasks=[9])
            trainer.fit(pl_model, data_1)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_1)
            # Second training
            # Callbacks
            early_stopping_callback = EarlyStopping(
                monitor=monitor, patience=patience, mode=mode)
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=ckpt_path,
                filename=name + f"-{args.name}" + "-{epoch:02d}",
            )
            lr_monitor_callback = LearningRateMonitor()
            trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=True, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])  # reload_dataloaders_every_n_epochs=1
            pl_model.model.train_z_only()
            trainer.fit(pl_model, data_2)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_2)
        elif special_training == "singlez":
            trainer.fit(pl_model, data)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data)
            # Second training
            # Callbacks
            early_stopping_callback = EarlyStopping(
                monitor=monitor, patience=patience, mode=mode)
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=ckpt_path,
                filename=name + f"-{args.name}" + "-{epoch:02d}",
            )
            lr_monitor_callback = LearningRateMonitor()
            trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=True, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])  # reload_dataloaders_every_n_epochs=1
            pl_model.model.train_z_only()
            pl_model.model.freeze_z()
            pl_model.model.train_special_z(mode="single")
            trainer.fit(pl_model, data)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data)
            # Third training
            # Callbacks
            early_stopping_callback = EarlyStopping(
                monitor=monitor, patience=patience, mode=mode)
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=ckpt_path,
                filename=name + f"-{args.name}" + "-{epoch:02d}",
            )
            lr_monitor_callback = LearningRateMonitor()
            trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=True, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])  # reload_dataloaders_every_n_epochs=1
            pl_model.model.train_z_only()
            pl_model.model.freeze_z()
            pl_model.model.train_special_z(mode="mean_z")
            trainer.fit(pl_model, data)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data)
        elif special_training == "singlezbis":
            pl_model.model.freeze_z()
            pl_model.model.train_special_z(mode="single")
            trainer.fit(pl_model, data)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data)
        elif special_training == "fewshot2":
            data_1 = LightningCifarTasks(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=2, tasks=[(0, 1), (2, 3)])
            data_2 = LightningCifarTasks(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=4, tasks=[(0, 2), (1, 3), (0, 3), (2, 1)], task_ids=[2, 3, 4, 5])
            trainer.fit(pl_model, data_1)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_1)
            # Second training
            # Callbacks
            early_stopping_callback = EarlyStopping(
                monitor=monitor, patience=patience, mode=mode)
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=ckpt_path,
                filename=name + f"-{args.name}" + "-{epoch:02d}",
            )
            lr_monitor_callback = LearningRateMonitor()
            trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=True, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])  # reload_dataloaders_every_n_epochs=1
            pl_model.model.train_z_only()
            trainer.fit(pl_model, data_2)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_2)
        elif special_training == "fewshot2bis":
            data_1 = LightningCifarTasks(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=6, tasks=[(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (2, 1)])
            trainer.fit(pl_model, data_1)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_1)
        elif special_training == "fewshot3":
            logger = TensorBoardLogger(save_dir=log_dir, name=name)
            csv_logger = CSVLogger(save_dir=csv_log_dir, name=name)
            data_1 = LightningCifarTasks(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=5, tasks=[(0,1), (2,3), (4,5), (6,7), (8,9)])
            data_2 = LightningCifarTasks(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=5, tasks=[(0,1), (2,3), (4,5), (6,7), (8,9)], color_jitter=True)
            trainer.fit(pl_model, data_1)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_1)
            # Second training
            logger = TensorBoardLogger(save_dir=log_dir+"bis", name=name)
            csv_logger = CSVLogger(save_dir=csv_log_dir+"bis", name=name)
            pl_model = LightningClassifierTask(model=model, batch_size=batch_size, patience=patience, monitor=monitor, mode=mode, lr_reduce_factor=lr_reduce_factor,
                                           latent_size=latent_size, learning_rate=learning_rate, use_sgd=use_sgd, lr_reduce=lr_reduce, use_optim=use_optim,
                                           batch_target_model=batch_target_model.__class__.__name__, hnet=hnet, input_type=input_type, encoder=encoder.__class__.__name__, batch=batch, sigma=sigma.item(), # next model params
                                           base=base, num_tasks=num_tasks, distribution=distribution, connectivity_type=connectivity_type, connectivity=connectivity, activation=activation, step=step, seed=seed,
                                           nbr_chunks=nbr_chunks, bias_sparse=bias, normalize=normalize, name=args.name, resnet_name=resnet_name, num_class_per_task=num_class_per_task, data=args.data, target_name=args.target, trials=args.trials, target_sparsity=target_sparsity)
            # Callbacks
            early_stopping_callback = EarlyStopping(
                monitor=monitor, patience=patience, mode=mode)
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=ckpt_path,
                filename=name + f"-{args.name}" + "-{epoch:02d}",
            )
            lr_monitor_callback = LearningRateMonitor()
            trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=True, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])
            trainer.fit(pl_model, data_2)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_2)
            # Third training
            logger = TensorBoardLogger(save_dir=log_dir+"ter", name=name)
            csv_logger = CSVLogger(save_dir=csv_log_dir+"ter", name=name)
            pl_model = LightningClassifierTask(model=model, batch_size=batch_size, patience=patience, monitor=monitor, mode=mode, lr_reduce_factor=lr_reduce_factor,
                                           latent_size=latent_size, learning_rate=learning_rate, use_sgd=use_sgd, lr_reduce=lr_reduce, use_optim=use_optim,
                                           batch_target_model=batch_target_model.__class__.__name__, hnet=hnet, input_type=input_type, encoder=encoder.__class__.__name__, batch=batch, sigma=sigma.item(), # next model params
                                           base=base, num_tasks=num_tasks, distribution=distribution, connectivity_type=connectivity_type, connectivity=connectivity, activation=activation, step=step, seed=seed,
                                           nbr_chunks=nbr_chunks, bias_sparse=bias, normalize=normalize, name=args.name, resnet_name=resnet_name, num_class_per_task=num_class_per_task, data=args.data, target_name=args.target, trials=args.trials, target_sparsity=target_sparsity)
            # Callbacks
            early_stopping_callback = EarlyStopping(
                monitor=monitor, patience=patience, mode=mode)
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=ckpt_path,
                filename=name + f"-{args.name}" + "-{epoch:02d}",
            )
            lr_monitor_callback = LearningRateMonitor()
            trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=True, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])
            trainer.fit(pl_model, data_1)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_2)
            # Fourth training
            logger = TensorBoardLogger(save_dir=log_dir+"quater", name=name)
            csv_logger = CSVLogger(save_dir=csv_log_dir+"quater", name=name)
            # Callbacks
            early_stopping_callback = EarlyStopping(
                monitor=monitor, patience=patience, mode=mode)
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=ckpt_path,
                filename=name + f"-{args.name}" + "-{epoch:02d}",
            )
            lr_monitor_callback = LearningRateMonitor()
            trainer = Trainer(fast_dev_run=fast_dev_run, max_epochs=max_epochs, enable_model_summary=True, gpus=1, auto_select_gpus=True, logger=[logger, csv_logger],
                          track_grad_norm=2, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, callbacks=[early_stopping_callback, lr_monitor_callback, checkpoint_callback])
            pl_model.model.train_z_only()
            trainer.fit(pl_model, data_2)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_2)
        elif special_training == "interp1-2":
            # cifar 10/5
            data_1 = LightningCifarTasks(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=2, tasks=[(0, 1), (2, 3)])
            trainer.fit(pl_model, data_1)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_1)
        elif special_training == "interp1-2bis":
            # cifar 10/5
            data_1 = LightningCifarTasks(batch_size=batch_size, num_class_per_task=num_class_per_task,
                              n_classes=n_classes, cifar=n_classes, num_tasks=5, tasks=[(0,1), (2,3), (4,5), (6,7), (8,9)])
            trainer.fit(pl_model, data_1)
            if not fast_dev_run:
                trainer.test(ckpt_path="best", dataloaders=data_1)
        else:
            raise ValueError()
