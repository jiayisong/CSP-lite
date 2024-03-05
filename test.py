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
    testset, _, cls, AP = dataset_init(only_test=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False,
                                             num_workers=8,  # collate_fn=augmentations.detection_collate,
                                             pin_memory=False)
    model = my_model(n_cls=cls)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print_parameter(model, summary_location)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    checkpoint = None

    for cp in os.listdir(summary_location):
        if cp == 'fuse_model.pth.tar' and os.path.isfile(summary_location + cp):
            print(f'loading checkpoint => {summary_location + cp}')
            checkpoint = torch.load(summary_location + cp)
            model.load_state_dict(checkpoint['state_dict'])
            print(f'loaded checkpoint => {summary_location + cp}')
    if checkpoint:
        result = val(testset, testloader, model, summary_location, None, AP)
    else:
        print('No checkpoint')


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


if __name__ == '__main__':
    main()
