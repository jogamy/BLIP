import os
import csv

from datasets import Dataset, Image
from datasets import load_dataset
import numpy as np

from ...datamodule import BaseDataset

class NICEtrainDataset(BaseDataset):
    def __init__(self, path, vis_processor, dec_tok, max_len):
        super().__init__(path, vis_processor, dec_tok, max_len)
        self.srcs, self.public_ids, self.tgts = self.load_image(path)

    def __getitem__(self, index):
        
        inp = self.vis_processor(self.srcs[index]['image'], return_tensors='np')

        inp['pixel_values'] = inp['pixel_values'].squeeze(0)
        
        ids = self.dec_tok.encode(self.tgts[index])
        
        return {
            'pixel_values' : inp['pixel_values'],
            'public_id' : self.public_ids[index],
            'caption': np.array(ids, dtype=np.int_),
        }
        
    def __len__(self):
        return len(self.srcs)
    
    def load_image(self, path):
        image_filename_list = os.listdir(os.path.join(path, 'val'))
        image_path_list = [os.path.join(path, 'val', image_filename) for image_filename in image_filename_list]
                
        ds = Dataset.from_dict({'image': image_path_list}).cast_column("image", Image())        

        image_filename_list = [image_filename.split(".")[0] for image_filename in image_filename_list]

        id_caption_dict = {}
        csv_path = os.path.join(path, 'nice-val-5k.csv')
        with open(csv_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                id_caption_dict[row['public_id']] = row['caption_gt']
        
        caption_list = []
        for image_filename in image_filename_list:
            caption_list.append(id_caption_dict[image_filename])    

        return ds, image_filename_list, caption_list

class NICEDataset(BaseDataset):
    def __init__(self, path, vis_processor, dec_tok, max_len):
        super().__init__(path, vis_processor, dec_tok, max_len)
        self.srcs, self.public_ids = self.load_image(path)

    def __getitem__(self, index):
        inp = self.vis_processor(self.srcs[index]['image'], return_tensors='np')
        inp['pixel_values'] = inp['pixel_values'].squeeze(0)
        inp['public_id'] = self.public_ids[index].split(".")[0]

        return {
            'pixel_values' : inp['pixel_values'],
            'public_id' : inp['public_id']
        }
        
    def __len__(self):
        return len(self.srcs)
    
    def load_image(self, path):

        image_filename_list = os.listdir(path)
        image_path_list = [os.path.join(path, image_filename) for image_filename in image_filename_list]
                
        ds = Dataset.from_dict({'image': image_path_list}).cast_column("image", Image())        

        return ds, image_filename_list