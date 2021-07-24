import torch
from transformers import BertTokenizer, BertModel
import os
import torch.nn as nn
from old.model_reference import Attention, BertIntermediate, BertOutput

UNCASED = './bert-base-uncased'
VOCAB = 'vocab.txt'


class ServeNet(torch.nn.Module):
    def __init__(self, hiddenSize):
        super(ServeNet, self).__init__()
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

        self.Liner = nn.Linear(self.hiddenSize, 360)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, names, descriptions):
        input_tokens_names, segment_ids_names, input_masks_names = names
        input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions

        name_bert_output = self.bert_name(input_tokens_names, segment_ids_names,
                                          input_masks_names)
        description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
                                                        input_masks_descriptions)

        # attention_name = self.name_self_attention(name_bert_output[0], name_bert_output[0])
        # name_BertIntermediate = self.name_BertIntermediate(attention_name)
        # name_BertOutput = self.name_BertOutput(name_BertIntermediate, name_bert_output[0])
        #
        # attention_description = self.description_self_attention(description_bert_output[0], description_bert_output[0])
        # description_BertIntermediate = self.description_BertIntermediate(attention_description)
        # description_BertOutput = self.description_BertOutput(description_BertIntermediate, description_bert_output[0])
        #

        name_bert_output0=name_bert_output[1].view(-1,1,768)
        name_bert_output1=name_bert_output[0]
        name_feature=torch.cat((name_bert_output0,name_bert_output1),1)

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
        # output = self.dropout(output)
        return output


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(UNCASED, VOCAB))

    name = tokenizer.encode_plus('web api', truncation=True, max_length=10,
                                 padding="max_length")
    description = tokenizer.encode_plus('hello，how are you,fine thank you and you', truncation=True, max_length=128,
                                        padding="max_length")

    input_ids_name = torch.tensor([name["input_ids"]]).cuda()
    segment_id_name = torch.tensor([name["token_type_ids"]]).cuda()
    input_mask_name = torch.tensor([name["attention_mask"]]).cuda()

    input_ids_descriptions = torch.tensor([description["input_ids"]]).cuda()
    segment_id_descriptions = torch.tensor([description["token_type_ids"]]).cuda()
    input_mask_descriptions = torch.tensor([description["attention_mask"]]).cuda()

    # print(input_ids_name.shape, segment_id_name.shape, input_mask_name.shape)
    # print(input_ids_descriptions.shape, segment_id_descriptions.shape, input_mask_descriptions.shape)

    model = ServeNet(768)
    model.cuda()

    y = model((input_ids_name, segment_id_name, input_mask_name,),
              (input_ids_descriptions, segment_id_descriptions, input_mask_descriptions))
    print(y.shape)
    # g = make_dot(y)
    # g.view()
    # summary(model, input_size=[(1, 1, 10), (1, 1, 10), (1, 1, 10), (1, 1, 128), (1, 1, 128), (1, 1, 128)])
    #
    #
    #

    # print(x.shape)

# class ServeNet(torch.nn.Module):
#     def __init__(self, hiddenSize):
#         super(ServeNet, self).__init__()
#         self.hiddenSize = hiddenSize
#         # cache_dir = None
#         self.bert_name = BertModel.from_pretrained(UNCASED)
#         self.bert_description = BertModel.from_pretrained(UNCASED)
#
#         self.name_self_attention = Attention(self.hiddenSize, self.hiddenSize, 8)
#         self.description_self_attention = Attention(self.hiddenSize, self.hiddenSize, 8)
#
#         self.name_BertIntermediate = BertIntermediate()
#         self.name_BertOutput = BertOutput()
#
#         self.description_BertIntermediate = BertIntermediate()
#         self.description_BertOutput = BertOutput()
#
#         self.Liner = nn.Linear((self.hiddenSize) * 2, 50)
#
#     def forward(self, names, descriptions):
#         input_tokens_names, segment_ids_names, input_masks_names = names
#         input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions
#
#         name_bert_output = self.bert_name(input_tokens_names, segment_ids_names,
#                                           input_masks_names)
#         description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
#                                                         input_masks_descriptions)
#         # self attention
#         attention_name = self.name_self_attention(name_bert_output[1], name_bert_output[1])
#         attention_description = self.description_self_attention(description_bert_output[1], description_bert_output[1])
#
#         name_BertIntermediate = self.name_BertIntermediate(attention_name)
#         name_BertOutput = self.name_BertOutput(name_BertIntermediate, name_bert_output[1])
#
#         description_BertIntermediate = self.description_BertIntermediate(attention_description)
#         description_BertOutput = self.description_BertOutput(description_BertIntermediate, description_bert_output[1])
#
#         # cross attention
#         attention_name = self.name_self_attention(name_BertOutput, description_BertOutput)
#         attention_description = self.description_self_attention(description_BertOutput, name_BertOutput)
#
#         name_BertIntermediate = self.name_BertIntermediate(attention_name)
#         name_BertOutput = self.name_BertOutput(name_BertIntermediate, name_BertOutput)
#
#         description_BertIntermediate = self.description_BertIntermediate(attention_description)
#         description_BertOutput = self.description_BertOutput(description_BertIntermediate, description_BertOutput)
#
#         # all_features = self.weight_sum(name_bert_output[1], description_bert_output[1])
#         output = self.Liner(torch.cat((description_BertOutput, name_BertOutput), 1))
#
#         return output


# class ServeNet(torch.nn.Module):
#     def __init__(self, hiddenSize):
#         super(ServeNet, self).__init__()
#         self.hiddenSize = hiddenSize
#         # cache_dir = None
#         self.bert_name = BertModel.from_pretrained(UNCASED)
#         self.bert_description = BertModel.from_pretrained(UNCASED)
#
#         self.name_self_attention=Attention(self.hiddenSize, self.hiddenSize, 8)
#         self.description_self_attention = Attention(self.hiddenSize, self.hiddenSize, 8)
#
#         self.name_BertIntermediate=BertIntermediate()
#         self.name_BertOutput = BertOutput()
#
#         self.description_BertIntermediate = BertIntermediate()
#         self.description_BertOutput = BertOutput()
#
#         self.Liner=nn.Linear(self.hiddenSize,50)
#
#
#     def forward(self, names, descriptions):
#         input_tokens_names, segment_ids_names, input_masks_names = names
#         input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions
#
#         name_bert_output = self.bert_name(input_tokens_names, segment_ids_names,
#                                           input_masks_names)
#         description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
#                                                         input_masks_descriptions)
#         # self attention
#         attention_name=self.name_self_attention(name_bert_output[1],name_bert_output[1])
#         attention_description=self.description_self_attention(description_bert_output[1],description_bert_output[1])
#
#         name_BertIntermediate=self.name_BertIntermediate(attention_name)
#         name_BertOutput = self.name_BertOutput(name_BertIntermediate,name_bert_output[1])
#
#         description_BertIntermediate = self.description_BertIntermediate(attention_description)
#         description_BertOutput = self.description_BertOutput(description_BertIntermediate, description_bert_output[1])
#
#         # cross attention
#         attention_name = self.name_self_attention(name_BertOutput, description_BertOutput)
#         attention_description = self.description_self_attention(description_BertOutput, name_BertOutput)
#
#         name_BertIntermediate = self.name_BertIntermediate(attention_name)
#         name_BertOutput = self.name_BertOutput(name_BertIntermediate, name_BertOutput)
#
#         description_BertIntermediate = self.description_BertIntermediate(attention_description)
#         description_BertOutput = self.description_BertOutput(description_BertIntermediate, description_BertOutput)
#
#
#         # all_features = self.weight_sum(name_bert_output[1], description_bert_output[1])
#         output = self.Liner(description_BertOutput)
#
#         return output


# class weighted_sum(nn.Module):
#     def __init__(self):
#         super(weighted_sum, self).__init__()
#         self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#
#
#     def forward(self, input1, input2):
#         return input1 * self.w1 + input2 * self.w2


# class ServeNet(torch.nn.Module):
#     def __init__(self, hiddenSize):
#         super(ServeNet, self).__init__()
#         self.hiddenSize = hiddenSize
#
#         self.bert_name = BertModel.from_pretrained(UNCASED)
#
#         self.bert_description = BertModel.from_pretrained(UNCASED)
#
#         self.name_liner = nn.Linear(in_features=self.hiddenSize, out_features=1024)
#         self.name_Relu = nn.ReLU()
#
#         # self.conv1 = nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3, 3), padding=0)
#         # self.conv2 = nn.Conv2d(in_channels=32,out_channels=1, kernel_size=(1, 1), padding=0)
#         # self.CNN_Dropout = nn.Dropout(p=0.1)
#         #
#         self.lstm = nn.LSTM(input_size=self.hiddenSize, hidden_size=512, num_layers=1, batch_first=True,
#                             bidirectional=True, dropout=0.1)
#
#         self.weight_sum = weighted_sum()
#         self.final_liner = nn.Linear(in_features=2048,out_features=50)
#
#     def forward(self, names, descriptions):
#         input_tokens_names, segment_ids_names, input_masks_names = names
#         input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions
#
#         # name
#
#         name_bert_output = self.bert_name(input_tokens_names, segment_ids_names,
#                                      input_masks_names)
#         name_features = self.name_liner(name_bert_output[1])
#         name_features = self.name_Relu(name_features)
#         # torch.Size([1, 1024])
#
#
#         description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
#                                                    input_masks_descriptions)
#         description_features = self.name_liner(description_bert_output[1])
#         description_features = self.name_Relu(description_features)
#
#
#
#         all_features=torch.cat((name_features,description_features),1)
#         output = self.final_liner(all_features)
#         output=F.softmax(output,dim=1)
#         return output


# raw
# class ServeNet(torch.nn.Module):
#     def __init__(self, hiddenSize):
#         super(ServeNet, self).__init__()
#         self.hiddenSize = hiddenSize
#
#         self.bert_name = BertModel.from_pretrained(UNCASED)
#
#         self.bert_description = BertModel.from_pretrained(UNCASED)
#
#         self.name_liner = nn.Linear(in_features=self.hiddenSize, out_features=1024)
#         self.name_ReLU = nn.ReLU()
#         self.name_Dropout = nn.Dropout(p=0.1)
#
#         # self.conv1 = nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3, 3), padding=0)
#         # self.conv2 = nn.Conv2d(in_channels=32,out_channels=1, kernel_size=(1, 1), padding=0)
#         # self.CNN_Dropout = nn.Dropout(p=0.1)
#         #
#         self.lstm = nn.LSTM(input_size=self.hiddenSize, hidden_size=512, num_layers=1, batch_first=True,
#                             bidirectional=True)
#
#         self.weight_sum = weighted_sum()
#         self.final_liner = nn.Linear(in_features=1024,out_features=50)
#
#     def forward(self, names, descriptions):
#         input_tokens_names, segment_ids_names, input_masks_names = names
#         input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions = descriptions
#
#         # name
#
#         name_bert_output = self.bert_name(input_tokens_names, segment_ids_names,
#                                      input_masks_names)
#         # Feature for Name
#         name_features = self.name_liner(name_bert_output[1])
#         name_features = self.name_ReLU(name_features)
#         name_features = self.name_Dropout(name_features)
#         # torch.Size([1, 1024])
#
#         # description
#
#
#         description_bert_output = self.bert_description(input_tokens_descriptions, segment_ids_descriptions,
#                                                    input_masks_descriptions)
#
#         # # LSTM
#         description_features = self.lstm(description_bert_output[0])
#         description_features = description_features[1][1].view(-1, 1024)
#
#         # sum
#         all_features = self.weight_sum(name_features, description_features)
#         output = self.final_liner(all_features)
#         return output
