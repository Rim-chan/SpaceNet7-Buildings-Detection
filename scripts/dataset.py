import torch
import numpy as np
import skimage.io as io
from utils import get_files
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split



class SpaceNet7(Dataset):
    def __init__(self, files, crop_size, exec_mode):
        
        self.files = files
        self.crop_size = crop_size
        self.exec_mode = exec_mode
    
    def OpenImage(self, idx, invert=True):
        image = io.imread(self.files[idx]['image'])[:,:,0:3] #shape (H, W, 3)
        if invert:
            image = image.transpose((2,0,1))                 #shape (3, H, W)
        return (image / np.iinfo(image.dtype).max) #render the values between 0 and 1
       
    
    def OpenMask(self, idx):
        mask = io.imread(self.files[idx]['mask'])
        return np.where(mask==255, 1, 0) #change the values to 0 and 1
       
    
    def __getitem__(self, idx):
        # read the images and masks as numpy arrays
        x = self.OpenImage(idx, invert=True)
        y = self.OpenMask(idx)
        # padd the images to have a homogenous size (C, 1024, 1024)
        x, y = self.padding((x,y[None]))
    
        # if it is the training phase, create random (C, 430, 430) crops
        # if it is the evaluation phase, we will leave the orginal size (C, 1024, 1024)
        if self.exec_mode =='train':
            x, y = self.crop(x[None], y[None], self.crop_size)
            x, y = x[0], y[0]
        
        # numpy array --> torch tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.uint8)
        
        # normalize the images (image- image.mean()/image.std())
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) 
        return normalize(x), y
    
    
    def __len__(self):
        return len(self.files)
    
    def padding (self, sample):
        image, mask = sample
        C, H, W = image.shape
        if H == 1024 and W == 1024:
            return image, mask
        
        if  H != 1024:
            image = np.pad(image, (((0,0), (1,0), (0,0))), 'constant', constant_values=(0))
            mask  = np.pad(mask,  (((0,0), (1,0), (0,0))), 'constant', constant_values=(0))
            
        if W != 1024:
            image = np.pad(image, (((0,0), (0,0), (1,0))), 'constant', constant_values=(0))
            mask  = np.pad(mask,  (((0,0), (0,0), (1,0))), 'constant', constant_values=(0))
        
        return image, mask

    
    def crop(self, data, seg, crop_size=256):
        data_shape = tuple([len(data)] + list(data[0].shape))
        data_dtype = data[0].dtype
        dim = len(data_shape) - 2

        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype
        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                             "dimensions. Data: %s, seg: %s" % \
                                                                             (str(data_shape), str(seg_shape))

        crop_size = [crop_size] * dim
        data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
        seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)


        for b in range(data_shape[0]):
            data_shape_here = [data_shape[0]] + list(data[b].shape)
            seg_shape_here = [[seg_shape[0]]] + list(seg[0].shape)

            lbs = []
            for i in range(len(data_shape_here) - 2):
                lbs.append(np.random.randint(0, data_shape_here[i+2] - crop_size[i]))

            ubs = [lbs[d] + crop_size[d] for d in range(dim)]

            slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            data_cropped = data[b][tuple(slicer_data)]

            slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            seg_cropped = seg[b][tuple(slicer_seg)]

            data_return[b] = data_cropped
            seg_return[b] = seg_cropped

        return data_return, seg_return


class SpaceNet7DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args  = args
        
    def setup(self, stage=None):
        files = get_files(self.args.base_dir)
        train_files, test_files = train_test_split(files, test_size=0.1, random_state=self.args.seed)
        self.spaceNet7_train = SpaceNet7(train_files, self.args.crop_size, self.args.exec_mode)
        self.spaceNet7_val = SpaceNet7(test_files, self.args.crop_size, self.args.exec_mode)

        
    def train_dataloader(self):
        train_sampler = self.ImageSampler(len(self.spaceNet7_train), self.args.samples_per_epoch)
        train_bSampler = BatchSampler(train_sampler, batch_size=self.args.batch_size, drop_last=True)
        return DataLoader(self.spaceNet7_train, batch_sampler=train_bSampler, num_workers=self.args.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.spaceNet7_val, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False)
    
    def predict_dataloader(self):
        return DataLoader(self.spaceNet7_val, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False)


    class ImageSampler(Sampler):
        def __init__(self, num_images=300, num_samples=500):
            self.num_images = num_images
            self.num_samples = num_samples

        def generate_iteration_list(self):
            return np.random.randint(0, self.num_images, self.num_samples)

        def __iter__(self):
            return iter(self.generate_iteration_list())

        def __len__(self):
            return self.num_samples
