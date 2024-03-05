
from config import *


def dataset_init(only_test=False):
    if args.dataset == 'caltech':
        return caltech_init(only_test)
    elif args.dataset == 'cityperson':
        return cityperson_init(only_test)
    else:
        raise RuntimeError('dataset name error')



def caltech_init(only_test):
    from datasets.my_caltech import CaltechDetection
    from test_tools.compute_MR import evaluation as MR
    if not only_test:
        trainset = CaltechDetection(root=caltech_root, image_set='train', input_size=test_size, down_factor=down_factor,
                                    training=True)
    else:
        trainset = None
    testset = CaltechDetection(root=caltech_root, image_set='test', input_size=test_size, training=False,
                               down_factor=down_factor)
    return testset, trainset, 1, MR



def cityperson_init(only_test):
    from datasets.my_cityperson import CityPersonDetection
    from test_tools.compute_MR import evaluation as MR
    if not only_test:
        trainset = CityPersonDetection(root=cityperson_root, image_set='train', input_size=test_size,
                                       down_factor=down_factor,
                                       training=True)
    else:
        trainset = None
    testset = CityPersonDetection(root=cityperson_root, image_set='val', input_size=test_size, training=False,
                                  down_factor=down_factor)
    return testset, trainset, 1, MR
