import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd


def load_data(file):
    pass #data loading func

class CelebADataset(Dataset):
    """CelebA Custom Dataset"""

    def __init__(self, root_dir, mode='train', target_attr='Arched_Eyebrows', image_size=224):
        """
        Args:
            root_dir (string): Directory with all the images and attribute files.
            mode (string): 'train', 'valid', or 'test' to select partition.
            target_attr (string): The target attribute for binary classification.
            image_size (int): The size to which images will be resized.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.image_dir = os.path.join(self.root_dir, 'img_align_celeba')
        
        # 속성 파일과 파티션 파일 경로 설정
        attr_path = os.path.join(self.root_dir, 'list_attr_celeba.txt')
        partition_path = os.path.join(self.root_dir, 'list_eval_partition.txt')

        # 속성 데이터 로드
        self.attr_df = pd.read_csv(attr_path, delim_whitespace=True, header=1, index_col=0)
        # 파티션 데이터 로드
        self.partition_df = pd.read_csv(partition_path, delim_whitespace=True, header=None, names=['image_id', 'partition'])

        # 파티션 코드로 데이터 필터링 (0: train, 1: valid, 2: test)
        partition_map = {'train': 0, 'valid': 1, 'test': 2}
        target_partition = partition_map[self.mode]
        self.partition_df = self.partition_df[self.partition_df['partition'] == target_partition]

        # 해당 파티션의 이미지 파일명과 속성 데이터만 남김
        self.file_list = self.partition_df['image_id'].tolist()
        self.attr_df = self.attr_df.loc[self.file_list]

        # 타겟 속성(라벨) 추출 및 변환 (-1, 1 -> 0, 1)
        self.labels = self.attr_df[target_attr].values
        self.labels[self.labels == -1] = 0
        
        # 이미지 변환 설정
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')
        # BCEWithLogitsLoss를 위해 라벨 타입을 float으로 변경
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        if self.transform:
            image = self.transform(image)
            
        return {'image': image, 'label': label}

class DataSet(Dataset):
    def __init__(self, file_list, label=None):
        self.file_list = file_list
        self.label = label

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.label is None:
            return {'data' : torch.tensor(load_data(self.file_list[index]), dtype=torch.float)}
        else:
            return {'data' : torch.tensor(load_data(self.file_list[index]), dtype=torch.float), 
                    'label' : torch.tensor(load_data(self.label[index]), dtype=torch.float)}
        
class Preload_DataSet(Dataset):
    def __init__(self, file_list, label=None):
        self.file_list = file_list
        self.label = label

        self.data = torch.stack([load_data[file] for file in file_list])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.label is None:
            return {'data' : self.data[index]}
        else:
            return {'data' : self.data[index], 
                    'label' : torch.tensor(load_data(self.label[index]), dtype=torch.float)}