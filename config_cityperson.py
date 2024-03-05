test_size = (1312, 2624)  # (h,w)
# test_size = (1024, 2048)  # (h,w)
train_size = (640, 1280)  # (h,w)
test_batch_size = 16
train_batch_size = 16
save_epochs = 1
epochs = 160
val_freq = 1
val_print_freq = 5
val_start_epoch = 200
cityperson_root = '/ssddata/DataSets/cityperson/'

def backbone_lr_lambda(epoch):
    return 2e-4

def neck_head_lr_lambda(epoch):
    return 2e-4
