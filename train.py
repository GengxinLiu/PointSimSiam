from encoder import pointnet2_ssg, dgcnn
import torch
from optimizers import get_optimizer, LR_Scheduler
from pretext import simsiam
from utils import view_bar, get_timestamp, show_acc_loss
from configs import configs_simsiam
from data import ModelNet40Unsup, RandomTransform, RandomCrop, ShapeNetUnsup, ShapeNetScanUnsup
from torch.utils.data import DataLoader
import numpy as np
from eval_linear import eval_encoder
import os
import torch.distributed as dist
import time
import shutil


def main():
    t_stamp = get_timestamp()
    log_name = 'log_simsiam'
    os.makedirs(f'{log_name}/{t_stamp}')
    os.makedirs(f'{log_name}/{t_stamp}/epochs')
    args = configs_simsiam()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    shutil.copy('run.sh', f'{log_name}/{t_stamp}')
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    # define dataloader
    if args.data == 'ModelNet40':
        data = ModelNet40Unsup(add_test=args.add_test)
    elif args.data == 'ShapeNet':
        data = ShapeNetUnsup(txt_file=args.txt)
    elif args.data == 'ShapeNetScan':
        data = ShapeNetScanUnsup(num_points=args.num_points, txt_file=args.txt, data_id=args.partial_id)
    train_loader = DataLoader(
        data, batch_size=args.batch_size, drop_last=True, shuffle=True
    )
    print(f"Unsupervised train data {args.data}, size {data.__len__()}")
    backbone = {
        'dgcnn': dgcnn.DGCNN(multi=args.multi, output_dim=args.output_dim),
        'pointnet2_ssg': pointnet2_ssg.pointnet2(multi=args.multi, output_dim=args.output_dim),
    }[args.backbone]

    model = simsiam.SimSiam(
        backbone=backbone, hid_dim_pro=args.project_hdim, out_dim_pro=args.project_odim, hid_dim_pre=args.predict_hdim
    )
    if args.proj_layers is not None:
        model.projector.set_layers(args.proj_layers)
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        dist.init_process_group(
            'gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    # define optimizer
    optimizer = get_optimizer(
        args.optimizer, model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    lr_scheduler = LR_Scheduler(
        optimizer=optimizer,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr * args.batch_size / 256,
        num_epochs=args.epochs,
        base_lr=args.lr,
        final_lr=args.final_lr * args.batch_size / 256,
        iter_per_epoch=len(train_loader),
        step_decay_lr=args.step_decay_lr,
        step_epochs=args.step_epochs,
        constant_predictor_lr=args.constant_predictor_lr  # see the end of section 4.2 predictor
    )

    # augmentation
    random_transform = RandomTransform(
        scale=args.scale, translate=args.translate, rotate_z=args.rotate_z, jitter=args.jitter,
        rotate_group=args.rotate_group,
        scale_low=args.scale_low, scale_high=args.scale_high,
        shift_range=args.shift_range, sigma=args.sigma
    )
    random_crop = RandomCrop(crop_size=args.crop_size, crop_norm=args.crop_norm)

    step = 0
    for epoch in range(args.epochs):
        if epoch % args.save == 0:
            torch.save(model.module.backbone,
                       f'{log_name}/{t_stamp}/epochs/{epoch}.pth')
        sum_loss = []
        model.train()
        torch.set_grad_enabled(True)
        start = time.time()
        for i, batch in enumerate(train_loader):
            if args.data == 'ShapeNetScan':
                view1, view2 = batch
                view2 = random_crop(random_transform(view2.numpy()))
            else:
                view1 = batch.numpy()
                view2 = random_crop(random_transform(batch.numpy()))
            model.zero_grad()
            loss = model.forward(view1.permute(0, 2, 1).float().to(device),
                                 torch.from_numpy(view2).permute(0, 2, 1).float().to(device))
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)  # multi gpus
            loss.backward()
            optimizer.step()
            lr = lr_scheduler.step()
            f = open(f'{log_name}/{t_stamp}/lr.txt', 'a+')
            f.write('{:.5f}\n'.format(lr))
            f.close()

            sum_loss.append(loss.cpu().item())
            log = 'epoch {:03d} step {:04d} time {:.0f}s loss {:.5f}'.format(
                epoch, step, time.time() - start, np.mean(sum_loss)
            )
            view_bar(log, i + 1, data.__len__() // args.batch_size)
            step += 1
        acc = eval_encoder(model.module.backbone, cls='svm',
                           train_ratio=0.05, test_ratio=1.0, verbose=False, seed=args.seed)
        log = 'epoch {:03d} step {:04d} time {:.0f}s loss {:.5f} acc {:.3f}'.format(
            epoch, step, time.time() - start, np.mean(sum_loss), acc
        )
        view_bar(log, i + 1, data.__len__() // args.batch_size)
        f = open(f'{log_name}/{t_stamp}/train.txt', 'a+')
        f.write(f'{log}\n')
        f.close()
        print()
        show_acc_loss(f'{log_name}/{t_stamp}/train.txt', f'{log_name}/{t_stamp}/loss_acc.png',
                      title=f'{log_name}/{t_stamp}')

        torch.save(model.module.backbone,
                   f'{log_name}/{t_stamp}/model.pth')
        torch.save(model.module, f'{log_name}/{t_stamp}/checkpoint.pth')


if __name__ == '__main__':
    main()
