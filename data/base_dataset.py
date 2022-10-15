import torch
from torch.utils.data import Dataset



class BaseDataset(Dataset):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """미니배치를 만들기 위해서 샘플 리스트를 머지합니다.

        Args:
            samples (List[dict]) : collate 할 샘플들
        
        Returns:
            dict: 모델에 포워딩하기 적합한 형태의 미니배치 
        
        """
        raise NotImplementedError