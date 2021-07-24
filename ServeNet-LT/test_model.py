import torch
from torch.utils.data import DataLoader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from data_pre import load_data_train, load_data_test


# model = ServeNet(768)
# model.load_state_dict(torch.load('./model/model.pkl'),False)
# model.load_state_dict(torch.load('./model/model.pkl'),False)


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
            label = torch.max(label, 1)[1]

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
            label = torch.max(label, 1)[1]
            outputs = model((input_tokens_name, segment_ids_name, input_masks_name),
                            (input_tokens_descriptions, segment_ids_descriptions, input_masks_descriptions))
            maxk = max((1, 5))
            y_resize = label.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            total += label.size(0)
            correct += torch.eq(pred, y_resize).sum().float().item()

    return 100 * correct / total


if __name__ == "__main__":
    train_data = load_data_train()
    test_data = load_data_test()
    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=32)
    test_dataloader = DataLoader(test_data, batch_size=32)
    model = torch.load("./model/training_max.pt")
    model.cuda()
    model.eval()
    evaluteTop1(train_dataloader, model, "train")
    evaluteTop1(test_dataloader, model, "test")
    evaluteTop5(train_dataloader, model, "train")
    evaluteTop5(test_dataloader, model, "test")
