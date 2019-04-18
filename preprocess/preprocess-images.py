import os
import sys
import h5py
import argparse
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import utils.config as config
import utils.data as data
import utils.utils as utils
from resnet import resnet as caffe_resnet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = caffe_resnet.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def create_coco_loader(*paths):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    datasets = [data.CocoImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', 
        default='0', 
        help='the gpu device id')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    net = Net().cuda()
    net.eval()

    loader = create_coco_loader(
        config.train_image_path, 
        config.val_image_path, 
        config.test_image_path)
    features_shape = (
        len(loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size)

    with h5py.File(config.preprocessed_path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        for ids, imgs in tqdm(loader):
            imgs = imgs.cuda(async=True)
            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            coco_ids[i:j] = ids.numpy().astype('int32')
            i = j


if __name__ == '__main__':
    main()
