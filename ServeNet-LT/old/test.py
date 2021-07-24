import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

f_train = open('data/50/BERT-ServiceDatasetWithNameMiniBatch-TrainData.pickle', 'rb')
f_test = open('data/50/BERT-ServiceDatasetWithNameMiniBatch-TestData.pickle', 'rb')
traindata = pickle.load(f_train)
testdata = pickle.load(f_test)
f_train.close()
f_test.close()
tranning_steps_per_epoch = len(traindata)
validation_steps = len(testdata)


def train_generator():
    for d in traindata:
        x_train = d[0]
        y_train = d[1]
        yield x_train, y_train



def test_generator():
    for d in testdata:
        x_test = d[0]
        y_test = d[1]
        yield x_test, y_test

if __name__=="__main__":
    index = 0
    for data, label in train_generator():
        print()
        print(index)
        print(label)
        index += 1

# for data, label in train_generator():
#     input_tokens_name = torch.tensor(data[3]).cuda()
#     segment_ids_name = torch.tensor(data[4]).cuda()
#     input_masks_name = torch.tensor(data[5]).cuda()
#
#     input_tokens_descriptions = torch.tensor(data[0]).cuda()
#     segment_ids_descriptions = torch.tensor(data[1]).cuda()
#     input_masks_descriptions = torch.tensor(data[2]).cuda()
#     print(label.shape)
#     print()

# df = pd.read_csv("data/50/train.csv")
# label = np.array(df.ServiceClassification)
# print(label)
#
#
#
# UNCASED = './bert-base-uncased'
# VOCAB = 'vocab.txt'
# tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
# train_data = load_data_train()
# # train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data,  batch_size=1)
#
# for batch_idx, data in enumerate(train_dataloader, 0):
#     input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions, input_tokens_name, segment_ids_name, input_masks_name, total_targets = data
#
#     print(tokenizer.convert_ids_to_tokens(input_tokens_name.numpy()[0]))
#     print(tokenizer.convert_ids_to_tokens(input_tokens_descriptions.numpy()[0]))
#     print(total_targets)
#     print()


#
# UNCASED = './bert-base-uncased'
# VOCAB = 'vocab.txt'
# tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
# bert_model = BertModel.from_pretrained(UNCASED)
#
# sen_code = tokenizer.encode_plus('hello，how are you', truncation=True,max_length=10,padding="max_length")
#
# print(sen_code)
#
# # 对编码进行转换，以便输入Tensor
# tokens_tensor = torch.tensor([sen_code['input_ids']])  # 添加batch维度并,转换为tensor,torch.Size([1, 19])
# segments_tensors = torch.tensor(sen_code['token_type_ids'])  # torch.Size([19])
#
# # bert_model.eval()
# # # 进行编码
# # with torch.no_grad():
# #     outputs = bert_model(input_ids=tokens_tensor, token_type_ids=segments_tensors,attention_mask=None)
# #     encoded_layers = outputs  # outputs类型为tuple
#
#     # print(encoded_layers[0].shape, encoded_layers[1].shape)
# # torch.Size([1, 19, 768]) torch.Size([1, 768])
# # torch.Size([1, 19, 768]) torch.Size([1, 12, 19, 19])
#
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # model = BertModel.from_pretrained("bert-base-uncased")
# # text = "Replace me by any text you'd like."
# # encoded_input = tokenizer(text, return_tensors='pt')
# # output = model(encoded_input)
# # print((output))
