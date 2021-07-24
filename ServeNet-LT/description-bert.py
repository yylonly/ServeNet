import pickle
import time

import torch
from torch.optim import lr_scheduler
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import os
from tqdm import tqdm
import torch.nn as nn
from Dataset import ServeNetDataset
from data_pre import load_data_train, load_data_test

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

UNCASED = './bert-base-uncased'
VOCAB = 'vocab.txt'
epochs = 40
SEED = 123
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
BATCH_SIZE=128
CLASS_NUM=360
# f_test = open('data/50/', 'rb')
def evaluteTop1(model, dataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataLoader:
            input_tokens_name = data[3].cuda()
            segment_ids_name = data[4].cuda()
            input_masks_name = data[5].cuda()

            input_tokens_descriptions = data[0].cuda()
            segment_ids_descriptions = data[1].cuda()
            input_masks_descriptions = data[2].cuda()
            label = data[6].cuda()

            outputs = model((input_tokens_name, segment_ids_name, input_masks_name),
                            (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))

            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return 100 * correct / total


def evaluteTop5(model, dataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataLoader:
            input_tokens_name = data[3].cuda()
            segment_ids_name = data[4].cuda()
            input_masks_name = data[5].cuda()

            input_tokens_descriptions = data[0].cuda()
            segment_ids_descriptions = data[1].cuda()
            input_masks_descriptions = data[2].cuda()
            label = data[6].cuda()
            outputs = model((input_tokens_name, segment_ids_name, input_masks_name),
                            (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))
            maxk = max((1, 5))
            y_resize = label.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            total += label.size(0)
            correct += torch.eq(pred, y_resize).sum().float().item()

    return 100 * correct / total

class ServeNet(torch.nn.Module):
    def __init__(self, hiddenSize,CLASS_NUM):
        super(ServeNet, self).__init__()
        self.hiddenSize = hiddenSize
        # cache_dir = None
        # self.bert_name = BertModel.from_pretrained(UNCASED)
        self.bert_description = BertModel.from_pretrained(UNCASED)
        self.Liner = nn.Linear(hiddenSize, CLASS_NUM)

    def forward(self, names, descriptions):
        input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions

        description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
                                                        input_masks_descriptions)
        output=self.Liner(description_bert_output[1])

        return output


if __name__ == "__main__":
    # train_dataset = ServeNetDataset('data/50/BERT-ServiceDatasetWithNameMiniBatch-TrainData.pickle')
    # test_dataset = ServeNetDataset('data/50/BERT-ServiceDatasetWithNameMiniBatch-TestData.pickle')
    # train_Dataloader = DataLoader(dataset=train_dataset,batch_size=1, shuffle=False)
    # test_Dataloader = DataLoader(dataset=test_dataset,batch_size=1,  shuffle=False)

    train_data = load_data_train(CLASS_NUM)
    test_data = load_data_test(CLASS_NUM)
    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


    model = ServeNet(768,CLASS_NUM)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    for epoch in range(epochs):
        print("Epoch:{},lr:{}".format(str(epoch+1),str(optimizer.state_dict()['param_groups'][0]['lr'])))
        scheduler.step()
        model.train()
        for data in tqdm(train_dataloader):

            input_tokens_name = data[3].cuda()
            segment_ids_name = data[4].cuda()
            input_masks_name = data[5].cuda()

            input_tokens_descriptions = data[0].cuda()
            segment_ids_descriptions = data[1].cuda()
            input_masks_descriptions = data[2].cuda()
            label = data[6].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model((input_tokens_name, segment_ids_name, input_masks_name),
                            (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        print("=======>top1 acc on the test:{}".format(str(evaluteTop1(model, test_dataloader))))
        print("=======>top5 acc on the test:{}".format(str(evaluteTop5(model, test_dataloader))))
