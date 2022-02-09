import os
import PIL
import numpy as np

from torch.utils.data import Dataset

class LFWDataset(Dataset):
    '''
        Dataset subclass for loading LFW images in PyTorch.
    '''
    def __init__(self, root_dir, pairs_file, transforms=None):
        self.lfw_dir = root_dir
        self.pairs_file = pairs_file

        self.read_pairs()
        self.get_paths()

        self.transforms = transforms

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img_file = self.path_list[index]
        img = PIL.Image.open(img_file)

        im_out = self.transforms(img)
        return im_out

    def read_pairs(self):
        '''Read LFW pairs file'''
        self.pairs = []
        with open(self.pairs_file, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                self.pairs.append(pair)
        self.pairs = np.array(self.pairs, dtype=tuple)

    def get_paths(self, file_ext='jpg'):
        nrof_skipped_pairs = 0
        self.path_list = []
        self.issame_list = []
        
        for pair in self.pairs:
            if len(pair) == 3:
                path0 = os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(self.lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                issame = False
            
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                self.path_list += (path0, path1)
                self.issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        
        if nrof_skipped_pairs > 0:
            print(f'Skipped {nrof_skipped_pairs} image pairs for {self.lfw_dir}')