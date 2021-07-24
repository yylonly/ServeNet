import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import os

from transformers import BertTokenizer

from old.common_function import get_bert_feature
des_max_length=200 #110#160
name_max_length=10
def get_train_data(catagory_num):
    train_file = "data/{}/train.csv".format(str(catagory_num))
    df = pd.read_csv(train_file)
    #servenet dataset
    values = np.array(df.ServiceClassification)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    UNCASED = './bert-base-uncased'
    VOCAB = 'vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))

    # descriptions
    descriptions = df["ServiceDescription"].tolist()
    input_ids_descriptions = [get_bert_feature(tokenizer, description, max_length=des_max_length)[0] for description in
                              descriptions]
    input_tokens_descriptions = torch.tensor(input_ids_descriptions)

    segment_id_descriptions = [get_bert_feature(tokenizer, description, max_length=des_max_length)[1] for description in
                               descriptions]
    segment_ids_descriptions = torch.tensor(segment_id_descriptions)

    input_mask_descriptions = [get_bert_feature(tokenizer, description, max_length=des_max_length)[2] for description in
                               descriptions]
    input_masks_descriptions = torch.tensor(input_mask_descriptions)

    # name
    ServiceNames = df["ServiceName"].tolist()
    input_ids_ServiceName = [get_bert_feature(tokenizer, ServiceName, max_length=name_max_length)[0] for ServiceName in ServiceNames]
    input_tokens_ServiceName = torch.tensor(input_ids_ServiceName)

    segment_id_ServiceName = [get_bert_feature(tokenizer, ServiceName, max_length=name_max_length)[1] for ServiceName in
                              ServiceNames]
    segment_ids_ServiceName = torch.tensor(segment_id_ServiceName)

    input_mask_ServiceName = [get_bert_feature(tokenizer, ServiceName, max_length=name_max_length)[2] for ServiceName in
                              ServiceNames]
    input_masks_ServiceName = torch.tensor(input_mask_ServiceName)

    total_targets = torch.tensor(integer_encoded)

    torch.save(input_tokens_descriptions, "data_tensor/{}/input_tokens_descriptions.pt".format(str(catagory_num)))
    torch.save(segment_ids_descriptions, "data_tensor/{}/segment_ids_descriptions.pt".format(str(catagory_num)))
    torch.save(input_masks_descriptions, "data_tensor/{}/input_masks_descriptions.pt".format(str(catagory_num)))

    torch.save(input_tokens_ServiceName, "data_tensor/{}/input_tokens_ServiceName.pt".format(str(catagory_num)))
    torch.save(segment_ids_ServiceName, "data_tensor/{}/segment_ids_ServiceName.pt".format(str(catagory_num)))
    torch.save(input_masks_ServiceName, "data_tensor/{}/input_masks_ServiceName.pt".format(str(catagory_num)))

    torch.save(total_targets, "data_tensor/{}/total_targets.pt".format(str(catagory_num)))

    # return train_data


def get_test_data(catagory_num):
    test_file = "data/{}/test.csv".format(str(catagory_num))
    df = pd.read_csv(test_file)
    values = np.array(df.ServiceClassification)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    UNCASED = './bert-base-uncased'
    VOCAB = 'vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))

    # descriptions
    descriptions = df["ServiceDescription"].tolist()
    input_ids_descriptions = [get_bert_feature(tokenizer, description, max_length=des_max_length)[0] for description in
                              descriptions]
    input_tokens_descriptions = torch.tensor(input_ids_descriptions)

    segment_id_descriptions = [get_bert_feature(tokenizer, description, max_length=des_max_length)[1] for description in
                               descriptions]
    segment_ids_descriptions = torch.tensor(segment_id_descriptions)

    input_mask_descriptions = [get_bert_feature(tokenizer, description, max_length=des_max_length)[2] for description in
                               descriptions]
    input_masks_descriptions = torch.tensor(input_mask_descriptions)

    # name
    ServiceNames = df["ServiceName"].tolist()
    input_ids_ServiceName = [get_bert_feature(tokenizer, ServiceName, max_length=name_max_length)[0] for ServiceName in ServiceNames]
    input_tokens_ServiceName = torch.tensor(input_ids_ServiceName)

    segment_id_ServiceName = [get_bert_feature(tokenizer, ServiceName, max_length=name_max_length)[1] for ServiceName in
                              ServiceNames]
    segment_ids_ServiceName = torch.tensor(segment_id_ServiceName)

    input_mask_ServiceName = [get_bert_feature(tokenizer, ServiceName, max_length=name_max_length)[2] for ServiceName in
                              ServiceNames]
    input_masks_ServiceName = torch.tensor(input_mask_ServiceName)

    total_targets = torch.tensor(integer_encoded)
    # total_targets=total_targets.view(-1,1)
    # print(total_targets)

    torch.save(input_tokens_descriptions, "data_test_tensor/{}/input_tokens_descriptions.pt".format(str(catagory_num)))
    torch.save(segment_ids_descriptions, "data_test_tensor/{}/segment_ids_descriptions.pt".format(str(catagory_num)))
    torch.save(input_masks_descriptions, "data_test_tensor/{}/input_masks_descriptions.pt".format(str(catagory_num)))

    torch.save(input_tokens_ServiceName, "data_test_tensor/{}/input_tokens_ServiceName.pt".format(str(catagory_num)))
    torch.save(segment_ids_ServiceName, "data_test_tensor/{}/segment_ids_ServiceName.pt".format(str(catagory_num)))
    torch.save(input_masks_ServiceName, "data_test_tensor/{}/input_masks_ServiceName.pt".format(str(catagory_num)) )

    torch.save(total_targets, "data_test_tensor/{}/total_targets.pt".format(str(catagory_num)))

    # return train_data


def load_data_train(catagory_num):
    input_tokens_descriptions = torch.load("data_tensor/{}/input_tokens_descriptions.pt".format(str(catagory_num)))
    segment_ids_descriptions = torch.load("data_tensor/{}/segment_ids_descriptions.pt".format(str(catagory_num)))
    input_masks_descriptions = torch.load("data_tensor/{}/input_masks_descriptions.pt".format(str(catagory_num)))

    input_tokens_ServiceName = torch.load("data_tensor/{}/input_tokens_ServiceName.pt".format(str(catagory_num)))
    segment_ids_ServiceName = torch.load("data_tensor/{}/segment_ids_ServiceName.pt".format(str(catagory_num)))
    input_masks_ServiceName = torch.load("data_tensor/{}/input_masks_ServiceName.pt".format(str(catagory_num)))

    total_targets = torch.load("data_tensor/{}/total_targets.pt".format(str(catagory_num)))
    train_data = TensorDataset(input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions,
                               input_tokens_ServiceName, segment_ids_ServiceName, input_masks_ServiceName,
                               total_targets)
    return train_data


def load_data_test(catagory_num):
    input_tokens_descriptions = torch.load("data_test_tensor/{}/input_tokens_descriptions.pt".format(str(catagory_num)))
    segment_ids_descriptions = torch.load("data_test_tensor/{}/segment_ids_descriptions.pt".format(str(catagory_num)))
    input_masks_descriptions = torch.load("data_test_tensor/{}/input_masks_descriptions.pt".format(str(catagory_num)))

    input_tokens_ServiceName = torch.load("data_test_tensor/{}/input_tokens_ServiceName.pt".format(str(catagory_num)))
    segment_ids_ServiceName = torch.load("data_test_tensor/{}/segment_ids_ServiceName.pt".format(str(catagory_num)))
    input_masks_ServiceName = torch.load("data_test_tensor/{}/input_masks_ServiceName.pt".format(str(catagory_num)))

    total_targets = torch.load("data_test_tensor/{}/total_targets.pt".format(str(catagory_num)))
    train_data = TensorDataset(input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions,
                               input_tokens_ServiceName, segment_ids_ServiceName, input_masks_ServiceName,
                               total_targets)
    return train_data




# if __name__=="__main__":
#     train_data = load_data_train(50)
#     train_dataloader = DataLoader(train_data, batch_size=1)
#     for i in train_dataloader:
#         print(i
#         print()

if __name__=="__main__":
    get_train_data(50)
    get_test_data(50)



# get_test_data()


# df = pd.read_csv("data/train.csv")
# values = np.array(df.ServiceClassification)
#
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(values)
# print(integer_encoded)
# print(label_encoder.classes_)


# train_data = get_data()


#

#
# for i, data in enumerate(train_dataloader):
#     print(data)
#     print()

# test1=[1,2,3,5,6]
#
# test=torch.tensor(test1)
# print(test)
# torch.save("./data_t")


# model=ServeNet()
# model.cuda()
# for i, (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions,input_tokens_ServiceName, segment_ids_ServiceName, input_masks_ServiceName, total_targets) in enumerate(train_dataloader):
#     # print("--------------------------------------------------------------------------------------")
#     # print(input_tokens_descriptions.shape, segment_ids_descriptions.shape, input_masks_descriptions.shape, total_targets.shape)
#     # print(input_tokens_ServiceName.shape, segment_ids_ServiceName.shape, input_masks_ServiceName.shape, total_targets.shape)
#     # print("--------------------------------------------------------------------------------------")
#     input_tokens_descriptions=input_tokens_descriptions.cuda()
#     segment_ids_descriptions=segment_ids_descriptions.cuda()
#     input_masks_descriptions=input_masks_descriptions.cuda()
#     x = model(input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions)
#     print(x.shape)
