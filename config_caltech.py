test_size = (480, 640)  # (h,w)
train_size = (384, 512)  # (h,w)
test_batch_size = 16
train_batch_size = 16
save_epochs = 1
epochs = 36
val_freq = 1
val_print_freq = 50
val_start_epoch = 0
caltech_root = '/ssddata/DataSets/caltech/'


# caltech_root = '/usr/jys/DataSets/caltech/'


def backbone_lr_lambda(epoch):
    return 1e-4


def neck_head_lr_lambda(epoch):
    return 1e-4
