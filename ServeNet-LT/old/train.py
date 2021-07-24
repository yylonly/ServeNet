import os
import torch

from old.test import train_generator
from test_model import evaluteTop1, evaluteTop5

from tensorboardX import SummaryWriter

# 定义Summary_Writer
writer = SummaryWriter('./tain_result')

SEED = 123
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from old.ServeNet import ServeNet
import torch.optim as optim
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())
# train_data = load_data_train(50)
# test_data = load_data_test(50)
# # train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
# test_dataloader = DataLoader(test_data, batch_size=32)

print("------------------data get finished-------------------------------")

model = ServeNet(768)

model = nn.DataParallel(model)
model = model.cuda()
model.train()

print("------------------model finished-------------------------------")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-6,
#                              weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1, alpha=0.9)
# optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, eps = EPSILON)

epochs = 40
# training steps 的数量: [number of batches] x [number of epochs].
# total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)








max_acc = 0.0
for epoch in range(epochs):  # loop over the dataset multiple times

    ave_loss = 0.0
    running_loss = 0.0
    # for batch_idx, data in enumerate(train_dataloader, 0):
    batch_idx=0
    for data, label in train_generator():
        # get the inputs; data is a list of [inputs, labels]
        input_tokens_name = torch.tensor(data[3]).cuda()
        segment_ids_name = torch.tensor(data[4]).cuda()
        input_masks_name = torch.tensor(data[5]).cuda()

        input_tokens_descriptions = torch.tensor(data[0]).cuda()
        segment_ids_descriptions = torch.tensor(data[1]).cuda()
        input_masks_descriptions = torch.tensor(data[2]).cuda()
        label=torch.tensor(label).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model((input_tokens_name, segment_ids_name, input_masks_name),
                        (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))

        loss = criterion(outputs, torch.max(label,1)[1])
        loss.backward()
        optimizer.step()
        # scheduler.step()
        running_loss += loss.item()
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1
        batch_idx+=1
        if (batch_idx + 1) % 10 == 0 :
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                epoch, batch_idx + 1, ave_loss))
    # current_acc1_train = evaluteTop1(model, "train")
    # current_acc5_train = evaluteTop5(model, "train")
    current_acc1 = evaluteTop1(model, "test")
    current_acc5 = evaluteTop5(model, "test")
#     # writer.add_scalars('top1', {"train": current_acc1_train, "test": current_acc1}, epoch)
#     # writer.add_scalars('top5', {"train": current_acc5_train, "test": current_acc5}, epoch)
#     # print("--------------epoch loss:{}".format(str(running_loss / (len(train_dataloader.dataset)))))
#     # writer.add_scalar('loss', running_loss / (len(train_dataloader.dataset) / BATCH_SIZE), epoch)
#
    if current_acc5 > max_acc:
        max_acc = current_acc5
        print(">>>>>> current top1 acc : {}".format(str(current_acc1)))
        print(">>>>>> current top5 acc : {}".format(str(current_acc5)))
        torch.save(model, "./model/training_max.pt")

torch.save(model, "./model/final_model.pt")
print('Finished Training')
model.eval()
evaluteTop1(model, "train")
evaluteTop1(model, "test")

evaluteTop5(model, "train")
evaluteTop5(model, "test")
# for i, (
# input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions, input_tokens_name, segment_ids_name,
# input_masks_name, total_targets) in enumerate(train_dataloader):
#     # print("--------------------------------------------------------------------------------------")
#     # print(input_tokens_descriptions.shape, segment_ids_descriptions.shape, input_masks_descriptions.shape, total_targets.shape)
#     # print(input_tokens_ServiceName.shape, segment_ids_ServiceName.shape, input_masks_ServiceName.shape, total_targets.shape)
#     # print("--------------------------------------------------------------------------------------")
#     input_tokens_name = input_tokens_name.cuda()
#     segment_ids_name = segment_ids_name.cuda()
#     input_masks_name = input_masks_name.cuda()
#
#     input_tokens_descriptions = input_tokens_descriptions.cuda()
#     segment_ids_descriptions = segment_ids_descriptions.cuda()
#     input_masks_descriptions = input_masks_descriptions.cuda()

#     x = model((input_tokens_name, segment_ids_name, input_masks_name),
#               (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))
#     print(x.shape)
