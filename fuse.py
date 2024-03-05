import math
import numpy as np
from models.build_model import my_model
from datasets.init_dataset import dataset_init
from config import *
import torch
from datetime import datetime
import os
from models.decode import decode as dc

def create_summary_path():
    summary_location = './summary/' + model_name + '-' + args.dataset + '-' + args.run_number + '/'
    if not os.path.exists(summary_location + 'mAP/'):
        os.makedirs(summary_location + 'mAP/')
    return summary_location


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
    summary_location = create_summary_path()
    testset, _, cls, AP = dataset_init()
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers,  # collate_fn=augmentations.detection_collate,
                                             pin_memory=pin_memory)
    model = my_model(n_cls=cls)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print_parameter(model, summary_location)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    checkpoint_list = []
    loss_list = []
    epoch_list = []
    if resume_epochs:
        for cp in os.listdir(summary_location):
            if cp[-8:] == '.pth.tar' and os.path.isfile(summary_location + cp):
                checkpoint = torch.load(summary_location + cp, map_location=torch.device('cpu'))
                if 'loss_result' in checkpoint.keys() and checkpoint['epoch'] < 3000:
                    loss_list.append(checkpoint['loss_result'][-1][0])
                    epoch_list.append(checkpoint['epoch'])
                    checkpoint_list.append(checkpoint['state_dict']['model1'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(summary_location + cp, checkpoint['epoch'] + 1))
    weight = []
    loss_list = [sum(loss_list) / len(loss_list) - i for i in loss_list]
    # loss_list = [(i - max(loss_list)) * (0 - 1) / (max(loss_list) - min(loss_list)) + 0 for i in loss_list]
    for e, l in zip(epoch_list, loss_list):
        weight.append(weight_compute(e, l))
    weight = [i / sum(weight) for i in weight]
    for i in sorted(range(len(epoch_list)), key=lambda k: epoch_list[k]):
        print(f'epoch:{epoch_list[i]} loss:{loss_list[i]} weight:{weight[i]}')
    with torch.no_grad():
        for name, parameters in model.named_parameters():
            p = torch.zeros_like(parameters)
            for alpha, x in zip(weight, checkpoint_list):
                if name in x.keys():
                    alpha_temp = x[name].cuda() * alpha
                else:
                    alpha_temp = x['module.' + name].cuda() * alpha
                p = p + alpha_temp
            parameters.data = p
        for name, buffers in model.named_buffers():
            if ('running_mean' in name) or ('running_var' in name):
                b = torch.zeros_like(buffers)
                for alpha, x in zip(weight, checkpoint_list):
                    if name in x.keys():
                        alpha_temp = x[name].cuda() * alpha
                    else:
                        alpha_temp = x['module.' + name].cuda() * alpha
                    b = b + alpha_temp
                buffers.data = b
    result = val(testset, testloader, model, summary_location, None, AP)
    # '''
    save_model({'state_dict': model.state_dict(), 'result': result}, summary_location, filename='fuse_model')
    # '''


def weight_compute(epoch, loss):
    # a = 149
    # if epoch > a:
    #    return 0
    # return 0.185 * (0.815 ** (a - 1 - epoch)) if epoch > 0 and epoch < a else (0.815 ** (a - 1 - epoch))
    return loss if loss > 0 and epoch >= 0 and epoch <= 1e8 else 0


def val(testset, test_loader, model, output_dir, iteration, AP):
    """Perform test on the test set"""

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

def save_model(state, summary_location, filename='checkpoint'):
    """Saves checkpoint to disk"""
    directory = summary_location
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename + '.pth.tar'
    torch.save(state, filename)


if __name__ == '__main__':
    main()
