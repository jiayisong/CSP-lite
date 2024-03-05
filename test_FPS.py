import time

import torchvision
from models.build_model import my_model as my_model
from datasets.init_dataset import dataset_init
from config import *
import torch
from datetime import datetime
import os
from fuse import create_summary_path
from models.decode import decode as dc


def print_parameter(model, summary_location):
    inputs = None
    if FLOPs or draw_graph:
        inputs = torch.zeros(1, 3, test_size[0], test_size[1])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
    if FLOPs:
        from thop import profile
        flops, params = profile(model, inputs=(inputs,), verbose=True)
        print('Number of model FLOPs: ' + str(flops))
        print('Number of model parameters: {}'.format(params))
    else:
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
    if draw_graph:
        from torchviz import make_dot
        graph_y = model(inputs)
        graph = make_dot(graph_y, params=dict(model.named_parameters()))
        graph.render(summary_location + 'model_graph', view=False)


def main():
    torch.cuda.synchronize()
    summary_location = create_summary_path()
    testset, _, cls, AP = dataset_init()
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                             num_workers=0, pin_memory=True)
    model = my_model(n_cls=cls)
    if torch.cuda.is_available():
        model = model.cuda()
    print_parameter(model, summary_location)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if resume_epochs:
        if os.path.isfile(summary_location + "fuse_model.pth.tar"):
            print("=> loading checkpoint '{}'".format(summary_location + "fuse_model.pth.tar"))
            checkpoint = torch.load(summary_location + "fuse_model.pth.tar")
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}'".format(summary_location + "fuse_model.pth.tar"))
            print(f'result{checkpoint["result"]}')
        else:
            print("=> no checkpoint found at '{}'".format(summary_location + "fuse_model.pth.tar"))
    val(testset, testloader, model)


def val(testset, testloader, model):
    """Perform test on the test set"""

    model.eval()
    if args.tensorrt:
        from torch2trt import torch2trt
        inputs = torch.zeros(1, 3, test_size[0], test_size[1]).cuda()
        model = torch2trt(model, [inputs])
    avg_spend_time = 0
    avg_num = 100
    warm_up = 10
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    with torch.no_grad():
        for i, img in enumerate(testloader):
            # for i in range(avg_num):
            # img = testset[i].unsqueeze(0)
            if i >= avg_num:
                break
            torch.cuda.synchronize()
            start_time = time.time()
            if use_cuda:
                img = img.cuda(non_blocking=True)
            pre = model(img)
            prediction = dc(pre)
            torch.cuda.synchronize()
            spend_time = time.time() - start_time
            # print(spend_time)
            if i >= warm_up:
                avg_spend_time += spend_time
        avg_spend_time = avg_spend_time / (avg_num - warm_up)
        FPS = 1 / avg_spend_time
        print(f'avg model time: {avg_spend_time} FPS: {FPS}')


def val2(testset, testloader, model):
    """Perform test on the test set"""

    model.eval()
    warm_up = 10
    with torch.no_grad():
        for i in range(warm_up + 1):
            img = testset[i].unsqueeze(0)

            if i >= warm_up:
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                ) as prof:
                    if torch.cuda.is_available():
                        img = img.cuda(non_blocking=True)
                    pre = model(img)
                    prediction = dc(pre)
                print(prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=-1))


if __name__ == '__main__':
    main()
