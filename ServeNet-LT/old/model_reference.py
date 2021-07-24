import torch.nn as nn
import torch
import math
import sys


class Attention(nn.Module):
    def __init__(self, input1_size, input2_size, num_attention_heads):
        super().__init__()
        if input1_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (input1_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(input1_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input1_size, self.all_head_size)
        self.key = nn.Linear(input2_size, self.all_head_size)
        self.value = nn.Linear(input2_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input1, input2, attention_mask=None):
        mixed_query_layer = self.query(input1)
        mixed_key_layer = self.key(input2)
        mixed_value_layer = self.value(input2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 3072)
        self.intermediate_act_fn = torch.nn.functional.relu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = torch.nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

#
# from numpy import unicode
#
#
# def gelu(x):
#     """Implementation of the gelu activation function.
#         For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
#         0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#         Also see https://arxiv.org/abs/1606.08415
#     """
#     return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
#
#
# class GeLU(nn.Module):
#     """Implementation of the gelu activation function.
#         For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
#         0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#         Also see https://arxiv.org/abs/1606.08415
#     """
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return gelu(x)
#
#
# def swish(x):
#     return x * torch.sigmoid(x)
#
#
# ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
# BertLayerNorm = torch.nn.LayerNorm
#
#
#
# class BertAttOutput(nn.Module):
#     def __init__(self, config):
#         super(BertAttOutput, self).__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states
# class BertSelfattLayer(nn.Module):
#     def __init__(self, config):
#         super(BertSelfattLayer, self).__init__()
#         self.self = BertAttention(config)
#         self.output = BertAttOutput(config)
#
#     def forward(self, input_tensor, attention_mask):
#         # Self attention attends to itself, thus keys and querys are the same (input_tensor).
#         self_output = self.self(input_tensor, input_tensor, attention_mask)
#         attention_output = self.output(self_output, input_tensor)
#         return attention_output
#
#
# class BertIntermediate(nn.Module):
#     def __init__(self, config):
#         super(BertIntermediate, self).__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act
#
#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states
# class BertOutput(nn.Module):
#     def __init__(self, config):
#         super(BertOutput, self).__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states
# class BertLayer(nn.Module):
#     def __init__(self, config):
#         super(BertLayer, self).__init__()
#         self.attention = BertSelfattLayer(config)
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)
#
#     def forward(self, hidden_states, attention_mask):
#         attention_output = self.attention(hidden_states, attention_mask)
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output
