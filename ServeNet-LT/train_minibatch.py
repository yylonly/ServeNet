import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import os
from tqdm import tqdm
import torch.nn as nn
from old.model_reference import Attention, BertIntermediate, BertOutput
from Dataset import ServeNetDataset

UNCASED = './bert-base-uncased'
VOCAB = 'vocab.txt'
epochs = 40
SEED = 123
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
BATCH_SIZE=128
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,7"


def evaluteTop1(model, dataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataLoader:
            input_tokens_name = data[3][0].cuda()
            segment_ids_name = data[4][0].cuda()
            input_masks_name = data[5][0].cuda()
            input_tokens_descriptions = data[0][0].cuda()
            segment_ids_descriptions = data[1][0].cuda()
            input_masks_descriptions = data[2][0].cuda()

            label = target[0].cuda()

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
        for data, target in dataLoader:
            input_tokens_name = data[3][0].cuda()
            segment_ids_name = data[4][0].cuda()
            input_masks_name = data[5][0].cuda()
            input_tokens_descriptions = data[0][0].cuda()
            segment_ids_descriptions = data[1][0].cuda()
            input_masks_descriptions = data[2][0].cuda()

            label = target[0].cuda()
            outputs = model((input_tokens_name, segment_ids_name, input_masks_name),
                            (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))
            maxk = max((1, 5))
            y_resize = label.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            total += label.size(0)
            correct += torch.eq(pred, y_resize).sum().float().item()

    return 100 * correct / total

class baseline(torch.nn.Module):
    def __init__(self, hiddenSize):
        super(baseline, self).__init__()
        self.hiddenSize = hiddenSize

        self.bert_name = BertModel.from_pretrained(UNCASED)

        self.bert_description = BertModel.from_pretrained(UNCASED)

        self.name_liner = nn.Linear(in_features=self.hiddenSize, out_features=1024)
        self.name_ReLU = nn.ReLU()
        self.name_Dropout = nn.Dropout(p=0.1)

        # self.conv1 = nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3, 3), padding=0)
        # self.conv2 = nn.Conv2d(in_channels=32,out_channels=1, kernel_size=(1, 1), padding=0)
        # self.CNN_Dropout = nn.Dropout(p=0.1)
        #
        self.lstm = nn.LSTM(input_size=self.hiddenSize, hidden_size=512, num_layers=1, batch_first=True,
                            bidirectional=True,dropout=0.1)

        self.weight_sum = weighted_sum()
        self.final_liner = nn.Linear(in_features=1024,out_features=50)

    def forward(self, names, descriptions):
        self.lstm.flatten_parameters()
        input_tokens_names, segment_ids_names, input_masks_names = names
        input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions

        # name

        name_bert_output = self.bert_name(input_tokens_names, segment_ids_names,
                                     input_masks_names)
        # Feature for Name
        name_features = self.name_liner(name_bert_output[1])
        name_features = self.name_ReLU(name_features)
        name_features = self.name_Dropout(name_features)
        # torch.Size([1, 1024])

        # description


        description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
                                                   input_masks_descriptions)

        # description_bert_output[0].shape：torch.Size([1, 128, 768])
        description_bert_feature=description_bert_output[0]

        # # LSTM
        packed_output, (hidden, cell) = self.lstm(description_bert_feature)
        hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)

        # sum
        all_features = self.weight_sum(name_features, hidden)
        output = self.final_liner(all_features)
        return output

class cross_attention_fc(torch.nn.Module):
    def __init__(self, hiddenSize):
        super(cross_attention_fc, self).__init__()
        self.hiddenSize = hiddenSize
        # cache_dir = None
        self.bert_name = BertModel.from_pretrained(UNCASED)
        self.bert_description = BertModel.from_pretrained(UNCASED)

        self.name_self_attention = Attention(self.hiddenSize, self.hiddenSize, 1)
        self.description_self_attention = Attention(self.hiddenSize, self.hiddenSize, 1)

        self.name_BertIntermediate = BertIntermediate()
        self.name_BertOutput = BertOutput()

        self.description_BertIntermediate = BertIntermediate()
        self.description_BertOutput = BertOutput()

        self.Liner = nn.Linear(self.hiddenSize, 50)

    def forward(self, names, descriptions):
        input_tokens_names, segment_ids_names, input_masks_names = names
        input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions

        name_bert_output = self.bert_name(input_tokens_names, segment_ids_names,
                                          input_masks_names)
        description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
                                                        input_masks_descriptions)

        name_bert_output0 = name_bert_output[1].view(-1, 1, 768)
        name_bert_output1 = name_bert_output[0]
        name_feature = torch.cat((name_bert_output0, name_bert_output1), 1)

        description_bert_output0 = description_bert_output[1].view(-1, 1, 768)
        description_bert_output1 = description_bert_output[0]
        description_feature = torch.cat((description_bert_output0, description_bert_output1), 1)

        # cross attention
        attention_name = self.name_self_attention(name_feature, description_feature)
        attention_description = self.description_self_attention(description_feature, name_feature)

        name_BertIntermediate = self.name_BertIntermediate(attention_name)
        name_BertOutput = self.name_BertOutput(name_BertIntermediate, name_feature)

        description_BertIntermediate = self.description_BertIntermediate(attention_description)
        description_BertOutput = self.description_BertOutput(description_BertIntermediate, description_feature)

        # # all_features = self.weight_sum(name_bert_output[1], description_bert_output[1])
        output = self.Liner(description_BertOutput[:, 0, :])
        return output
class weighted_sum(nn.Module):
    def __init__(self):
        super(weighted_sum, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)


    def forward(self, input1, input2):
        return input1 * self.w1 + input2 * self.w2
class cross_attention_lstm_ws_fc(torch.nn.Module):
    def __init__(self, hiddenSize):
        super(cross_attention_lstm_ws_fc, self).__init__()
        self.hiddenSize = hiddenSize
        # cache_dir = None
        self.bert_name = BertModel.from_pretrained(UNCASED)
        self.bert_description = BertModel.from_pretrained(UNCASED)

        self.name_self_attention = Attention(self.hiddenSize, self.hiddenSize, 1)
        self.description_self_attention = Attention(self.hiddenSize, self.hiddenSize, 1)

        self.name_BertIntermediate = BertIntermediate()
        self.name_BertOutput = BertOutput()

        self.description_BertIntermediate = BertIntermediate()
        self.description_BertOutput = BertOutput()

        self.name_lstm=nn.LSTM(input_size=self.hiddenSize, hidden_size=512, num_layers=1, batch_first=True,
                            bidirectional=True,dropout=0.1)
        self.description_lstm = nn.LSTM(input_size=self.hiddenSize, hidden_size=512, num_layers=1, batch_first=True,
                                 bidirectional=True, dropout=0.1)

        self.weight_sum = weighted_sum()
        self.Liner = nn.Linear(1024, 50)

    def forward(self, names, descriptions):
        self.name_lstm.flatten_parameters()
        self.description_lstm.flatten_parameters()
        input_tokens_names, segment_ids_names, input_masks_names = names
        input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions

        name_bert_output = self.bert_name(input_tokens_names, segment_ids_names,
                                          input_masks_names)
        description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
                                                        input_masks_descriptions)

        name_bert_output0 = name_bert_output[1].view(-1, 1, 768)
        name_bert_output1 = name_bert_output[0]
        name_feature = torch.cat((name_bert_output0, name_bert_output1), 1)

        description_bert_output0 = description_bert_output[1].view(-1, 1, 768)
        description_bert_output1 = description_bert_output[0]
        description_feature = torch.cat((description_bert_output0, description_bert_output1), 1)

        # cross attention
        attention_name = self.name_self_attention(name_feature, description_feature)
        attention_description = self.description_self_attention(description_feature, name_feature)

        name_BertIntermediate = self.name_BertIntermediate(attention_name)
        name_BertOutput = self.name_BertOutput(name_BertIntermediate, name_feature)

        description_BertIntermediate = self.description_BertIntermediate(attention_description)
        description_BertOutput = self.description_BertOutput(description_BertIntermediate, description_feature)

        name_lstm_output,(hidden_name,cell_name)=self.name_lstm(name_BertOutput)
        description_lstm_output,(hidden_description,cell_description) = self.description_lstm(description_BertOutput)

        hidden_name = torch.cat((hidden_name[0, :, :], hidden_name[1, :, :]), dim=1)
        hidden_description = torch.cat((hidden_description[0, :, :], hidden_description[1, :, :]), dim=1)


        all_features = self.weight_sum(hidden_name, hidden_description)
        output = self.Liner(all_features)
        return output

if __name__=="__main__":
    train_dataset = ServeNetDataset('data/50/BERT-ServiceDatasetWithNameMiniBatch-TrainData-length160.pickle')
    test_dataset = ServeNetDataset('data/50/BERT-ServiceDatasetWithNameMiniBatch-TestData-length160.pickle')
    train_Dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    test_Dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = baseline(768)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)


    for epoch in range(epochs):
        print("Epoch:{},lr:{}".format(str(epoch + 1), str(optimizer.state_dict()['param_groups'][0]['lr'])))
        scheduler.step()
        model.train()
        for data, target in tqdm(train_Dataloader):
            input_tokens_name = data[3][0].cuda()
            segment_ids_name = data[4][0].cuda()
            input_masks_name = data[5][0].cuda()
            input_tokens_descriptions = data[0][0].cuda()
            segment_ids_descriptions = data[1][0].cuda()
            input_masks_descriptions = data[2][0].cuda()

            label=target[0].cuda()

            optimizer.zero_grad()
            outputs = model((input_tokens_name, segment_ids_name, input_masks_name),
                                                    (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        print("=======>top1 acc on the test:{}".format(str(evaluteTop1(model, test_Dataloader))))
        print("=======>top5 acc on the test:{}".format(str(evaluteTop5(model, test_Dataloader))))








# max_acc = 0.0
# for epoch in range(epochs):  # loop over the dataset multiple times
#
#     ave_loss = 0.0
#     running_loss = 0.0
#     # for batch_idx, data in enumerate(train_dataloader, 0):
#     batch_idx=0
#     for data, label in train_generator():
#         # get the inputs; data is a list of [inputs, labels]
#         input_tokens_name = torch.tensor(data[3]).cuda()
#         segment_ids_name = torch.tensor(data[4]).cuda()
#         input_masks_name = torch.tensor(data[5]).cuda()
#
#         input_tokens_descriptions = torch.tensor(data[0]).cuda()
#         segment_ids_descriptions = torch.tensor(data[1]).cuda()
#         input_masks_descriptions = torch.tensor(data[2]).cuda()
#         label=torch.tensor(label).cuda()
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = model((input_tokens_name, segment_ids_name, input_masks_name),
#                         (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))
#
#         loss = criterion(outputs, torch.max(label,1)[1])
#         loss.backward()
#         optimizer.step()
#         # scheduler.step()
#         running_loss += loss.item()
#         ave_loss = ave_loss * 0.9 + loss.item() * 0.1
#         batch_idx+=1
#         if (batch_idx + 1) % 10 == 0 :
#             print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
#                 epoch, batch_idx + 1, ave_loss))
#     # current_acc1_train = evaluteTop1(model, "train")
#     # current_acc5_train = evaluteTop5(model, "train")
#     current_acc1 = evaluteTop1(model, "test")
#     current_acc5 = evaluteTop5(model, "test")
# #     # writer.add_scalars('top1', {"train": current_acc1_train, "test": current_acc1}, epoch)
# #     # writer.add_scalars('top5', {"train": current_acc5_train, "test": current_acc5}, epoch)
# #     # print("--------------epoch loss:{}".format(str(running_loss / (len(train_dataloader.dataset)))))
# #     # writer.add_scalar('loss', running_loss / (len(train_dataloader.dataset) / BATCH_SIZE), epoch)
# #
#     if current_acc5 > max_acc:
#         max_acc = current_acc5
#         print(">>>>>> current top1 acc : {}".format(str(current_acc1)))
#         print(">>>>>> current top5 acc : {}".format(str(current_acc5)))
#         torch.save(model, "./model/training_max.pt")
#
# torch.save(model, "./model/final_model.pt")
# print('Finished Training')
# model.eval()
# evaluteTop1(model, "train")
# evaluteTop1(model, "test")
#
# evaluteTop5(model, "train")
# evaluteTop5(model, "test")
# # for i, (
# # input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions, input_tokens_name, segment_ids_name,
# # input_masks_name, total_targets) in enumerate(train_dataloader):
# #     # print("--------------------------------------------------------------------------------------")
# #     # print(input_tokens_descriptions.shape, segment_ids_descriptions.shape, input_masks_descriptions.shape, total_targets.shape)
# #     # print(input_tokens_ServiceName.shape, segment_ids_ServiceName.shape, input_masks_ServiceName.shape, total_targets.shape)
# #     # print("--------------------------------------------------------------------------------------")
# #     input_tokens_name = input_tokens_name.cuda()
# #     segment_ids_name = segment_ids_name.cuda()
# #     input_masks_name = input_masks_name.cuda()
# #
# #     input_tokens_descriptions = input_tokens_descriptions.cuda()
# #     segment_ids_descriptions = segment_ids_descriptions.cuda()
# #     input_masks_descriptions = input_masks_descriptions.cuda()
#
# #     x = model((input_tokens_name, segment_ids_name, input_masks_name),
# #               (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))
# #     print(x.shape)
