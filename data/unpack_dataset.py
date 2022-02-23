import numbers
import argparse
import pickle
import cv2
import torch
import os
import mxnet as mx
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

class MXFaceDataset(Dataset):
    def __init__(self, root_dir):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),])

        self.root_dir = root_dir

        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

    def get_num_classes(self):
        return max(self.imgidx)

def unpack_train_dataset(src_dir, dst_dir):
    data_iter = MXFaceDataset(src_dir)

    i = {}
    for sample, label in tqdm(data_iter):
        label_dir = os.path.join(dst_dir, str(label.item()))
        
        if not os.path.isdir(label_dir):
            os.mkdir(label_dir)

        if label.item() not in i:
            i[label.item()] = 0
        else:
            i[label.item()] += 1

        filename = os.path.join(label_dir, f'{i[label.item()]}.jpg')
        if not os.path.isfile(filename):
            sample.save(filename)

def unpack_validation_dataset(src_path, image_size, dst_dir):
    try:
        with open(src_path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(src_path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    
    if isinstance(issame_list[0], int):
        issame_list = [num == 1 for num in issame_list]

    pairs_count = {}
    for idx in tqdm(range(0, len(issame_list)*2, 2)):
        _bin = bins[idx]
        img1 = mx.image.imdecode(_bin)
        _bin = bins[idx+1]
        img2 = mx.image.imdecode(_bin)

        pair_label = f'{issame_list[idx//2]}'

        if pair_label not in pairs_count:
            pairs_count[pair_label] = 0

        pair_number = pairs_count[pair_label]
        pairs_count[pair_label] += 1

        if img1.shape[1] != image_size[0]:
            img1 = mx.image.resize_short(img1, image_size[0])
        if img2.shape[1] != image_size[0]:
            img2 = mx.image.resize_short(img2, image_size[0])

        np_img1 = img1.asnumpy()
        np_img1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)

        np_img2 = img2.asnumpy()
        np_img2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(dst_dir, f'{pair_label}_pair_{pair_number}_1.jpg'), np_img1)
        cv2.imwrite(os.path.join(dst_dir, f'{pair_label}_pair_{pair_number}_2.jpg'), np_img2)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--subset', choices=['train', 'validation'], required=True,
                    help='Whether you are unpacking a training or a validation dataset')
    ap.add_argument('--train-src', metavar='PATH',
                    help='Path to the source dir where the .rec and .idx files are stored')
    ap.add_argument('--val-src', metavar='PATH',
                    help='Path to the .bin file')
    ap.add_argument('--dst', required=True, metavar='PATH',
                    help='Path to the dir where the unpacked .jpg images will be saved')                    
    ap.add_argument('--val-image-size', default=112, type=int,
                    help='The validation image size, it will be set to (size, size)')

    args = vars(ap.parse_args())

    if args['subset'] == 'train':
        unpack_train_dataset(args['train_src'], args['dst'])
    else:
        unpack_validation_dataset(args['val_src'], (args['val_image_size'], args['val_image_size']), args['dst'])