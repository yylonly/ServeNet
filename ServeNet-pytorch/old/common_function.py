


def get_bert_feature(tokenizer, sentence, max_length):
    sen_code = tokenizer.encode_plus(sentence, max_length=max_length,  truncation=True,padding="max_length")
    return sen_code['input_ids'],sen_code['token_type_ids'],sen_code['attention_mask']