
import argparse
import os
import cv2
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm


from src.utils.distribute_dispath import resume_training , load_checkpoint, save_checkpoint

from src.utils.load_config import load_config_module, parse_from_module
from src.utils.training_utils import naive_to_cuda, update_avg, avg_losses, forward_on_cuda, mp_run_on_node, dist_print
from src.utils.log import TensorBoardLogger

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


@torch.no_grad()
def val(model, dataset, loss, gpu, reduce=True, tb_logger=None):
    model.eval()
    t = tqdm(total=len(dataset) + 1) if gpu == 0 else None
    losses_avg = {}
    for idx, (input_data, gt_data,_,_,_) in enumerate(dataset):
        # Ignore loss for backward in validation
     
        _, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, model)

        update_avg(losses_avg, loss_iter)
        if t:
            # Now is the all-reduced batch loss.
            losses_show = loss_iter
            t.set_postfix(losses_show)
            t.update()
    # Global mean loss.
    losses_show = avg_losses(losses_avg)
    # rank = dist.get_rank()
    if reduce:
        for k, v in losses_show.items():
            v = torch.tensor(v).cuda()
            dist.reduce(v, 0)
            # dist.reduce(v, 0, op=ReduceOp.AVG)  # send to 0th slave
            # No need to all-reduce here. and if AVG works, will drop codes below
            if dist.get_rank() == 0:
                losses_show[k] = v.item() / dist.get_world_size()
    if t:
        if tb_logger:
            tb_logger.log_epoch(losses_show, prefix="val")
        t.set_postfix(losses_show)
        t.update()
        t.close()

def train_epoch(model, dataset, loss, optimizer, gpu, show_mean=False, tb_logger=None, scheduler=None):
    # Last iter as mean loss of whole epoch
    model.train()
    t = tqdm(total=len(dataset) + 1) if gpu == 0 else None
    losses_avg = {}
    for idx, (input_data, gt_data,_,_,_) in enumerate(dataset):
        # print(input_data.shape)
        
        loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, model)
        update_avg(losses_avg, loss_iter)

        optimizer.zero_grad()
        # torch.cuda.synchronize()
        # print("OPT zero DONE")
        # scaler.scale(loss_back).backward()
        loss_back.backward()
        
        # torch.cuda.synchronize()
        # print("Loss backward DONE")
     
        # scaler.step(optimizer)
        optimizer.step()
        # scaler.update()

        scheduler.step()

        # torch.cuda.synchronize()
        # print("OPT step DONE")
        if t:
            if show_mean:
                losses_show = avg_losses(losses_avg)
            else:
                losses_show = loss_iter
            if tb_logger:
                tb_logger.log_iter(losses_show, prefix="train")
            t.set_postfix(losses_show)
            t.update()
    if t:
        losses_show = avg_losses(losses_avg)
        if tb_logger:
            tb_logger.log_epoch(losses_show, prefix="train")
        t.set_postfix(losses_show)
        t.update()
        t.close()


def train_single_task(local_rank, world_size, config_file, checkpoint_path, eval_only):

 
    rank =  local_rank
    

    configs = load_config_module(config_file)
    attrs = parse_from_module(configs)

    dp = attrs.distribute_params
    print("{} -> {}\t{}/{}".format(dp.backend, dp.dist_url, rank, world_size))
    dist.init_process_group(backend=dp.backend, init_method=dp.dist_url, world_size=world_size, rank=rank)
   
    torch.cuda.set_device(local_rank)

    model = attrs.model()
    
    if torch.cuda.is_available():
        model = model.cuda(local_rank)

    # TODO: params template
    if attrs.use_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # model_info = ModelInfoTemplate(attrs.print_model, attrs.print_summary, attrs.input_size)
    # model_info = model_info(model)
    # if model_info:
    #     dist_print(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=attrs.find_unused_parameters)

    attrs.to_device()

    loss = attrs.loss

    Dataset = attrs.dataset

    train_dataset = Dataset('train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    dist_print("Training size: {}".format(len(train_dataset)))
    kwargs = {
        "sampler": train_sampler,
    }
    train_loader = DataLoader(train_dataset, **kwargs, **attrs.loader_args, pin_memory=True)

    # TODO: params template
    optimizer = attrs.optimizer(model.parameters(), **attrs.optimizer_params)
    scheduler = attrs.scheduler(optimizer,steps_per_epoch=len(train_loader), **attrs.scheuler_param)

    resume_epoch =0
    if checkpoint_path:
        if attrs.load_optimizer:
            resume_epoch =  resume_training(checkpoint_path, model.module, optimizer, scheduler)
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    # model._sync_params_and_buffers()  # Make sure the model is in sync with the other workers.


    if attrs.with_validation:
        val_dataset = Dataset('val')
        dist_print("Validation size: {}".format(len(val_dataset)))
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        kwargs = {
            "sampler": val_sampler,
        }
        val_loader = DataLoader(val_dataset, **kwargs, **attrs.val_loader_args, pin_memory=True)
        # val_loss = attrs.val_loss # getattr(configs, "val_loss", loss)
        if eval_only:
            val(model, val_loader, attrs.val_loss, local_rank)
            return

    # tensorboard_logger = getattr(configs, "tensorboard_log", None)
    tensorboard_logger = attrs.tensorboard_log if dist.get_rank() == 0 else None
    if attrs.tensorboard_log:
        tb_logger = TensorBoardLogger(tensorboard_logger)
    else:
        tb_logger = None

    for epoch in range(resume_epoch+1, attrs.epochs):
        dist_print("Training epoch = {}".format(epoch))
        train_sampler.set_epoch(epoch)
        train_epoch(model, train_loader, loss, optimizer, local_rank, tb_logger, scheduler=scheduler)
        # scheduler.step()
        if attrs.with_validation:
            dist_print("Validation @epoch {}".format(epoch))
            val(model, val_loader, attrs.val_loss, local_rank, tb_logger=tb_logger)
        # TODO benchmark
        # if epoch % 5 == 0:
        save_checkpoint(model, optimizer, attrs.save_path, 'ep%03d.pth' % epoch, epoch=epoch)
        # Drop optimizer
        # TODO template config file.
        save_checkpoint(model, None, attrs.save_path, 'latest.pth', epoch=epoch)




if __name__ == '__main__':

    # import sys
    # sys.path.append()
    # os.environ['CUDA_VISIBLE_DEVICES']='2'
    

    parser = argparse.ArgumentParser(description='PyTorch HDMap Training')
    parser.add_argument('config', metavar='DIR', help='path to config module')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    args = parser.parse_args()
    cp_path = args.resume


    # python tools/single_task_train.py config configs/multi_cam.py
    torch.cuda.empty_cache()
    mp_run_on_node(args.config, cp_path, args.evaluate, worker=train_single_task)
    