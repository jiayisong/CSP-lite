import random
import numpy as np
from config import *
from models.build_model import my_model
from datasets.init_dataset import dataset_init
import torch
from datetime import datetime
import os
from models.compute_loss import Loss as cpl
from models.decode import decode as dc
import fuse, test_FPS
from fuse import create_summary_path, print_parameter
import torch.distributed as dist
import torch.multiprocessing as mp


def generate_random_size():
    if mul_size_train is not None:
        min_size = mul_size_train[0] // 32
        max_size = mul_size_train[1] // 32
        may_size = []
        for w in range(min_size, max_size + 1):
            for h in range(min_size, max_size + 1):
                if w < 2 * h and h < 2 * w:
                    may_size.append((h * 32, w * 32))
    else:
        may_size = [train_size]
    return may_size


def set_random_seed():
    if seed is not None:
        random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def get_random_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())


def set_random_state(state):
    random_state, np_random_state, torch_random_state, torch_cuda_random_state = state
    random.setstate(random_state)
    np.random.set_state(np_random_state)
    torch.set_rng_state(torch_random_state)
    torch.cuda.set_rng_state(torch_cuda_random_state)


def main(proc):
    set_random_seed()
    loss_name = ['', '', '', '', '', '']
    loss_num = len(loss_name)
    start_epoch = 0
    loss_result = []
    map_result = []
    summary_location = create_summary_path()
    if tensorboard and proc == 0:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter(summary_location[2:])
    else:
        tensorboard_writer = None
    print(f'proc {proc} => preparing dataset ...')
    testset, trainset, cls, AP = dataset_init()
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers,  # collate_fn=augmentations.detection_collate,
                                             pin_memory=pin_memory)
    if len(args.gpu) > 1:
        print(f'proc {proc} => init process group ...')
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:10010',
                                world_size=len(args.gpu),
                                rank=proc)
        print(f'proc {proc} => process group is ok!...')
        torch.cuda.set_device(proc)
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, seed=seed)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size // len(args.gpu),
                                                  pin_memory=pin_memory, sampler=trainsampler,
                                                  num_workers=num_workers)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                                  pin_memory=pin_memory, num_workers=num_workers)
    print(f'proc {proc} => dataset is ok!')
    print(f'proc {proc} => preparing model ...')
    model = model_helper(cls, proc)
    print(f'proc {proc} => model is ok!')
    if proc == 0:
        print_parameter(model.model1, summary_location)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    if args.mixed_precision_training and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    criterion = cpl()
    trainsize = generate_random_size()
    if len(args.gpu) > 1:
        base_params = list(map(id, model.model1.module.backbone.parameters()))
    else:
        base_params = list(map(id, model.model1.backbone.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.model1.parameters())
    base_params = filter(lambda p: id(p) in base_params, model.model1.parameters())
    params = [
        {"params": base_params, "lr": init_lr},
        {"params": logits_params, "lr": init_lr}
    ]
    if optm == 'SGD':
        optimizer = torch.optim.SGD(params, lr=init_lr,
                                    momentum=0.9, weight_decay=weight_decay,
                                    nesterov=True)
    elif optm == 'adam':
        optimizer = torch.optim.Adam(params, lr=init_lr)
    else:
        raise RuntimeError('optimizer error')
    if resume_epochs:
        if os.path.islink(summary_location + "checkpoint.pth.tar"):
            # if True:
            checkpoint_file = os.readlink(summary_location + "checkpoint.pth.tar")
            # checkpoint_file = '/usr/jys/CSP-lite/summary/CSP-caltech-122/epoch[24].pth.tar'
            if os.path.isfile(checkpoint_file):
                print("=> loading checkpoint '{}'".format(checkpoint_file))
                if torch.cuda.is_available():
                    checkpoint = torch.load(checkpoint_file)
                else:
                    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
                loss_result, map_result, start_epoch = checkpoint['loss_result'], checkpoint['map_result'], checkpoint[
                    'epoch'] + 1
                # save_result(np.array(loss_result).transpose([1, 0]).tolist(), 'losses', summary_location)
                # save_result(np.array(map_result).transpose([1, 0]).tolist(), 'mAPs', summary_location)
                model.load_state_dict(checkpoint['state_dict'])
                set_random_state(checkpoint['random_state'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if scaler is not None:
                    scaler.load_state_dict(checkpoint['scaler'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_file, checkpoint['epoch'] + 1))
            else:
                print("=> no checkpoint found at '{}'".format(summary_location + "checkpoint.pth.tar"))
        else:
            print("=> no checkpoint found at '{}'".format(summary_location + "checkpoint.pth.tar"))
    # model.load_state_dict({'model1':(torch.load(summary_location + "fuse_model.pth.tar"))['state_dict']})
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[backbone_lr_lambda, neck_head_lr_lambda],
                                                  last_epoch=start_epoch - 1)
    if val_start_epoch == -1 and proc == 0:
        val(testset, testloader, model, summary_location, None, AP)
    for i in range(start_epoch, epochs):
        # train for one epoch
        if len(args.gpu) > 1:
            trainsampler.set_epoch(i)
        random_size = random.choice(trainsize)
        trainset.set_train_size(random_size)
        print('train size is %d * %d' % (random_size[0], random_size[1]))
        avg_losses = train(trainloader, model, criterion, optimizer, i, tensorboard_writer, scheduler, loss_num,
                           loss_name, scaler)
        scheduler.step(epoch=None)
        loss_result.append(avg_losses)
        # evaluate on validation set
        if proc == 0:
            print('\n' + "%s:BigStep[%d]" % (datetime.now(), i), end=' ')
            for loss_i in range(loss_num):
                print('%s:%f' % (loss_name[loss_i], avg_losses[loss_i]), end=' ')
            print('\n')

            save_result(np.array(loss_result).transpose([1, 0]).tolist(), 'losses', summary_location)
            if i >= val_start_epoch and ((1 + i) % val_freq) == 0:
                map_val = val(testset, testloader, model, summary_location, i, AP)
                map_result.append(map_val)
                save_result(np.array(map_result).transpose([1, 0]).tolist(), 'mAPs', summary_location)
            if save_epochs > 0 and ((1 + i) % save_epochs) == 0:
                save_model({'epoch': i, 'state_dict': model.state_dict(),
                            'loss_result': loss_result,
                            'map_result': map_result,
                            'random_state': get_random_state(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict() if scaler is not None else None}, summary_location, i, None)


def train(train_loader, model, criterion, optimizer, epoch, tensorboard_writer, scheduler, loss_num, loss_name, scaler):
    """Train for one epoch on the training set"""
    # switch to train mode

    losses = [AverageMeter() for _ in range(loss_num)]
    model.model1.train()
    loss = [0 for _ in range(loss_num)]
    for i, img_target in enumerate(train_loader):
        if torch.cuda.is_available():
            img = img_target[0].cuda(non_blocking=True)
            targets = []
            for target in img_target[1:]:
                targets.append(target.cuda(non_blocking=True))
        else:
            img = img_target[0]
            targets = img_target[1:]
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pre = model.model1(img)
                loss_ = criterion(pre, targets)
            scaler.scale(loss_[0]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pre = model.model1(img)
            loss_ = criterion(pre, targets)
            loss_[0].backward()
            optimizer.step()  # 反向传播，更新网络参数
        if torch.isinf(loss_[0]).item() or torch.isnan(loss_[0]).item():
            print("%s:Step[%d][%d]" % (datetime.now(), epoch, i), end=' ')
            for loss_i in range(loss_num):
                print('%s:%f' % (
                    loss_name[loss_i],
                    loss_[loss_i].item() if isinstance(loss_[loss_i], torch.Tensor) else loss_[loss_i]),
                      end=' ')
            raise RuntimeError('loss is nan or inf')

        for loss_i in range(loss_num):
            losses[loss_i].update(
                (loss_[loss_i].item() if isinstance(loss_[loss_i], torch.Tensor) else loss_[loss_i]) * img.size(0),
                img.size(0))
            loss[loss_i] += (loss_[loss_i].item() if isinstance(loss_[loss_i], torch.Tensor) else loss_[loss_i])

        if tensorboard:
            for loss_i in range(loss_num):
                tensorboard_writer.add_scalar(loss_name[loss_i], loss[loss_i], epoch * len(train_loader) + i)
        optimizer.zero_grad()  # 清空梯度
        model.filter_weight()
        if i % train_print_freq == 0:
            print("%s:Step[%d][%d]" % (datetime.now(), epoch, i), end=' ')
            for loss_i in range(loss_num):
                print('%s:%f' % (loss_name[loss_i], loss[loss_i]), end=' ')
            print("learn_rate:" + str(scheduler.get_last_lr()))
        loss = [0 for _ in range(loss_num)]
    return [losses[i].avg for i in range(loss_num)]


def val(testset, test_loader, model, output_dir, iteration, AP):
    """Perform test on the test set"""

    model = model.get_val_model()
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, img in enumerate(test_loader):
            if torch.cuda.is_available():
                img = img.cuda(non_blocking=True)
            pre = model(img)
            prediction = dc(pre)
            predictions += prediction
            if i % val_print_freq == 0:
                print("%s:Step[%d/%d]  rate_of_progress:%.2f%%" % (
                    datetime.now(), i, len(test_loader), (i + 1) * 100.0 / len(test_loader)))
        map_val = AP(testset, predictions, output_dir, test_size, down_factor, iteration)
        return map_val


def save_result(result, names, summary_location):
    file = open(summary_location + names + '.txt', 'w')
    file.write(str(result))
    file.close()


def save_model(state, summary_location, epoch, iteration):
    """Saves checkpoint to disk"""
    directory = summary_location
    if not os.path.exists(directory):
        os.makedirs(directory)
    if iteration is None:
        filename = f'{directory}epoch[{epoch}].pth.tar'
    else:
        filename = f'{directory}epoch[{epoch}][{iteration}].pth.tar'
    torch.save(state, filename)
    if os.path.islink(directory + 'checkpoint.pth.tar'):
        os.unlink(directory + 'checkpoint.pth.tar')
    os.symlink(filename, directory + 'checkpoint.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, vals, nums):
        self.sum += vals
        self.count += nums
        self.avg = self.sum * 1.0 / self.count


class model_helper(object):
    def __init__(self, cls, proc):
        self.alpha = filt_weight_alpha[::-1]
        self.beta = filt_weight_beta[:0:-1]
        self.beta0 = filt_weight_beta[0]
        self.filter = args.filt_weight
        self.model1 = my_model(n_cls=cls)
        self.model1.eval()
        self.fs = filt_weight_fs
        self.sample_number = 0
        # for name, value in self.model1.named_parameters():
        #    print('name: {0}.\t grad: {1}'.format(name, value.requires_grad))
        if torch.cuda.is_available():
            self.model1.cuda()
            if len(args.gpu) > 1:
                self.model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model1)
                self.model1 = torch.nn.parallel.DistributedDataParallel(self.model1, device_ids=[proc],
                                                                        output_device=proc,
                                                                        #  find_unused_parameters=True
                                                                        )
        if self.filter and proc == 0:
            random_state = get_random_state()
            with torch.no_grad():
                self.parameters_and_buffers = {'x': [], 'y': []}
                for _ in self.alpha:
                    temp = {}
                    for name, parameters in self.model1.named_parameters():
                        temp[name] = parameters.clone()
                    for name, buffers in self.model1.named_buffers():
                        if ('running_mean' in name) or ('running_var' in name):
                            temp[name] = buffers.clone()
                    self.parameters_and_buffers['x'].append(temp)
                for _ in self.beta:
                    temp = {}
                    for name, parameters in self.model1.named_parameters():
                        temp[name] = parameters.clone()
                    for name, buffers in self.model1.named_buffers():
                        if ('running_mean' in name) or ('running_var' in name):
                            temp[name] = buffers.clone()
                    self.parameters_and_buffers['y'].append(temp)
                self.model2 = my_model(n_cls=cls)
                self.model2.eval()
                if torch.cuda.is_available():
                    self.model2.cuda()
                for name, parameters in self.model2.named_parameters():
                    parameters.data = self.parameters_and_buffers['x'][0][name]
                for name, buffers in self.model2.named_buffers():
                    if ('running_mean' in name) or ('running_var' in name):
                        buffers.data = self.parameters_and_buffers['x'][0][name]
            set_random_state(random_state)

    def load_state_dict(self, checkpoints):
        self.model1.load_state_dict(checkpoints['model1'])
        if self.filter:
            self.parameters_and_buffers = checkpoints['parameters_and_buffers']
            self.sample_number = checkpoints['sample_number']
            if len(self.parameters_and_buffers['y']) > 0:
                for name, parameters in self.model2.named_parameters():
                    parameters.data = self.parameters_and_buffers['y'][-1][name]
                for name, buffers in self.model2.named_buffers():
                    if ('running_mean' in name) or ('running_var' in name):
                        buffers.data = self.parameters_and_buffers['y'][-1][name]

    def state_dict(self):
        sd = {'model1': self.model1.state_dict()}
        if self.filter:
            sd['parameters_and_buffers'] = self.parameters_and_buffers
            sd['sample_number'] = self.sample_number
        return sd

    def get_val_model(self):
        if self.filter:
            return self.model2
        else:
            return self.model1

    def filter_weight(self):
        if self.filter:
            if (self.sample_number % self.fs) == 0:
                self.sample_number = 0
                with torch.no_grad():
                    self.parameters_and_buffers['x'].pop(0)
                    temp = {}
                    for name, parameters in self.model1.named_parameters():
                        temp[name] = parameters.clone()
                    for name, buffers in self.model1.named_buffers():
                        if ('running_mean' in name) or ('running_var' in name):
                            temp[name] = buffers.clone()
                    self.parameters_and_buffers['x'].append(temp)

                    for name, parameters in self.model2.named_parameters():
                        if 'backbone' in name:
                            p = torch.zeros_like(parameters)
                            for alpha, x in zip(self.alpha, self.parameters_and_buffers['x']):
                                alpha_temp = alpha * x[name]
                                p = p + alpha_temp
                                # if name == 'backbone.resnet.layer4.2.bn3.weight':
                                #    print('x',x[name][-3].item())
                                #    print('alpha*x',alpha_temp[-3].item())
                                #    print(p[-3].item())
                            for beta, y in zip(self.beta, self.parameters_and_buffers['y']):
                                beta_temp = beta * y[name]
                                p = p - beta_temp
                                # if name == 'backbone.resnet.layer4.2.bn3.weight':
                                #    print('y', y[name][-3].item())
                                #    print('beta*y',beta_temp[-3].item())
                                #    print(p[-3].item())
                            p = p / self.beta0
                            # if name == 'backbone.resnet.layer4.2.bn3.weight':
                            #    print(p[-3].item())
                        else:
                            p = self.parameters_and_buffers['x'][-1][name]
                        # if name == 'backbone.resnet.layer4.2.bn3.weight':
                        #    print(self.parameters_and_buffers['x'][-1][name][-3].item(),end='=>')
                        #    print(p[-3].item())
                        parameters.data = p

                    for name, buffers in self.model2.named_buffers():
                        if 'backbone' in name:
                            if ('running_mean' in name) or ('running_var' in name):
                                b = torch.zeros_like(buffers)
                                for alpha, x in zip(self.alpha, self.parameters_and_buffers['x']):
                                    b = b + alpha * x[name]
                                for beta, y in zip(self.beta, self.parameters_and_buffers['y']):
                                    b = b - beta * y[name]
                                b = b / self.beta0
                                buffers.data = b
                        else:
                            if ('running_mean' in name) or ('running_var' in name):
                                b = self.parameters_and_buffers['x'][-1][name]
                                buffers.data = b
                    if len(self.parameters_and_buffers['y']) > 0:
                        self.parameters_and_buffers['y'].pop(0)
                        temp = {}
                        for name, parameters in self.model2.named_parameters():
                            temp[name] = parameters.clone()
                        for name, buffers in self.model2.named_buffers():
                            if ('running_mean' in name) or ('running_var' in name):
                                temp[name] = buffers.clone()
                        self.parameters_and_buffers['y'].append(temp)
            self.sample_number += 1


if __name__ == '__main__':
    if len(args.gpu) > 1:
        mp.spawn(main, nprocs=len(args.gpu), args=())
    else:
        main(0)
    fuse.main()
    test_FPS.main()
