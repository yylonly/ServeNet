import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ServeNetDataset(Dataset):
    def __init__(self,data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

if __name__=="__main__":
    dataset=ServeNetDataset('data/50/BERT-ServiceDatasetWithNameMiniBatch-TrainData.pickle')
    data_loader = DataLoader(dataset=dataset, shuffle=False)
    for data, label in tqdm(data_loader):
        print(data[0][0])
        print(data[0][0].shape,data[1][0].shape,data[2][0].shape,label[0].shape)
        print(data[3][0].shape, data[4][0].shape, data[5][0].shape, label[0].shape)