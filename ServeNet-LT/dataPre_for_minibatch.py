import numpy as np
import pandas as pd
from transformers import BertTokenizer
import os
import sklearn.preprocessing as pre_processing
import numpy as np
UNCASED = './bert-base-uncased'
VOCAB = 'vocab.txt'
BATCH_SIZE=128
MAXDESLENGTH=160



def prepareData_BERT_MiniBatch(dataframe, batchsize, maxDesLength, maxNameLength=15):
    resultData = []

    for i, batch_group in dataframe.groupby(np.arange(len(dataframe)) // batchsize):
        print("batch:",i)
        maxLengthInMiniBatch = batch_group.iloc[-1].ServiceDescriptionLength

        batch_name_list = batch_group["ServiceName"].tolist()

        name_input_ids_list = [
            tokenizer.encode_plus(x, max_length=maxNameLength, padding="max_length", truncation=True)["input_ids"]
            for x in batch_name_list]
        name_token_type_ids_list = [
            tokenizer.encode_plus(x, max_length=maxNameLength, padding="max_length", truncation=True)[
                "token_type_ids"]
            for x in batch_name_list]
        name_attention_mask_list = [
            tokenizer.encode_plus(x, max_length=maxNameLength, padding="max_length", truncation=True)[
                "attention_mask"]
            for x in batch_name_list]

        batch_description_list = batch_group["ServiceDescription"].tolist()
        if maxLengthInMiniBatch >= maxDesLength:
            description_input_ids_list = [
                tokenizer.encode_plus(x, max_length=maxDesLength, padding="max_length", truncation=True)["input_ids"]
                for x in batch_description_list]
            description_token_type_ids_list = [
                tokenizer.encode_plus(x, max_length=maxDesLength, padding="max_length", truncation=True)[
                    "token_type_ids"]
                for x in batch_description_list]
            description_attention_mask_list = [
                tokenizer.encode_plus(x, max_length=maxDesLength, padding="max_length", truncation=True)[
                    "attention_mask"]
                for x in batch_description_list]
        else:
            description_input_ids_list = [
                tokenizer.encode_plus(x, max_length=maxLengthInMiniBatch, padding="max_length", truncation=True)[
                    "input_ids"]
                for x in batch_description_list]
            description_token_type_ids_list = [
                tokenizer.encode_plus(x, max_length=maxLengthInMiniBatch, padding="max_length", truncation=True)[
                    "token_type_ids"]
                for x in batch_description_list]
            description_attention_mask_list = [
                tokenizer.encode_plus(x, max_length=maxLengthInMiniBatch, padding="max_length", truncation=True)[
                    "attention_mask"]
                for x in batch_description_list]

        labels = batch_group["labels"].tolist()
        data = [np.array(description_input_ids_list), np.array(description_token_type_ids_list), np.array(description_attention_mask_list), np.array(name_input_ids_list), np.array(name_token_type_ids_list), np.array(name_attention_mask_list)], np.array(labels)
        resultData.append(data)
    return resultData



if __name__ == "__main__":
    train = pd.read_csv("data/50/train_source.csv")
    test = pd.read_csv("data/50/test_source.csv")


    label = pre_processing.LabelEncoder()
    labels = label.fit_transform(train["ServiceClassification"].values)
    train["labels"]=labels

    label_test = pre_processing.LabelEncoder()
    labels_test = label_test.fit_transform(test["ServiceClassification"].values)
    test["labels"] = labels_test

    tokenizer = BertTokenizer.from_pretrained(os.path.join(UNCASED, VOCAB))
    train_data=prepareData_BERT_MiniBatch(train,batchsize=BATCH_SIZE,maxDesLength=MAXDESLENGTH)
    test_data = prepareData_BERT_MiniBatch(test,batchsize=BATCH_SIZE,maxDesLength=MAXDESLENGTH)

    import pickle

    f = open('data/50/BERT-ServiceDatasetWithNameMiniBatch-TrainData-length{}.pickle'.format(MAXDESLENGTH), 'wb')
    pickle.dump(train_data, f)
    f.close()

    f = open('data/50/BERT-ServiceDatasetWithNameMiniBatch-TestData-length{}.pickle'.format(MAXDESLENGTH), 'wb')
    pickle.dump(test_data, f)
    f.close()



    # test1_InputExamples = g.apply(lambda x: bert.run_classifier.InputExample(guid=None,
    #                                                                          text_a=x['ServiceDescription'],
    #                                                                          text_b=None,
    #                                                                          label=x[LABEL_COLUMN]), axis=1)
    #
    # maxLengthInMiniBatch = len(test1_InputExamples.iloc[-1].text_a)
    # print("MiniBatch MaxLength: " + str(maxLengthInMiniBatch))

    # test2_InputExamples = g.apply(lambda x: bert.run_classifier.InputExample(guid=None,
    #                                                                          text_a=x['ServiceName'],
    #                                                                          text_b=None,
    #                                                                          label=x[LABEL_COLUMN]), axis=1)
    #
    # #     # We'll set sequences to be at most 128 tokens long.
    #
    # # # Convert our train and test features to InputFeatures that BERT understands.
    # if maxLengthInMiniBatch >= maxDesLength:
    #     test1_features = bert.run_classifier.convert_examples_to_features(test1_InputExamples, label_list,
    #                                                                       maxDesLength, tokenizer)
    # else:
    #     test1_features = bert.run_classifier.convert_examples_to_features(test1_InputExamples, label_list,
    #                                                                       maxLengthInMiniBatch, tokenizer)
    #
    # test2_features = bert.run_classifier.convert_examples_to_features(test2_InputExamples, label_list,
    #                                                                   maxNameLength, tokenizer)
    #
    # X1_test = np.array([o.input_ids for o in test1_features])
    # X1_mask_test = np.array([o.input_mask for o in test1_features])
    # X1_segment_test = np.array([o.segment_ids for o in test1_features])
    #
    # X2_test = np.array([o.input_ids for o in test2_features])
    # X2_mask_test = np.array([o.input_mask for o in test2_features])
    # X2_segment_test = np.array([o.segment_ids for o in test2_features])
    #
    # Y_test = np.array([o.label_id for o in test2_features])
    # Test_Y_one_hot = convert_to_one_hot(Y_test, 50)
    #
    # data = [X1_test, X1_mask_test, X1_segment_test, X2_test, X2_mask_test, X2_segment_test], Test_Y_one_hot
    #
    # resultData.append(data)

    # return resultData
