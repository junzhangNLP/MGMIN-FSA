import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from tools.kan import KAN

class LanguageEmbeddingLayer(nn.Module):
    def __init__(self, model_type='chinese_bert', model_path=None):
        super(LanguageEmbeddingLayer, self).__init__()
        self.model_type = model_type
        
        if model_path is None:
            if model_type == 'chinese_bert':
                model_path = ''
            elif model_type == 'english_roberta':
                model_path = ''
        
        if model_type == 'chinese_bert':
            self.bertmodel = AutoModel.from_pretrained(model_path)
            self.use_token_type = True
        elif model_type == 'english_roberta':
            self.bertmodel = AutoModel.from_pretrained(model_path)
            self.use_token_type = False
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, bert_sent, bert_sent_mask, bert_sent_type=None):
        if self.use_token_type and bert_sent_type is not None:
            bert_output = self.bertmodel(
                input_ids=bert_sent,
                attention_mask=bert_sent_mask,
                token_type_ids=bert_sent_type
            )
        else:
            bert_output = self.bertmodel(
                input_ids=bert_sent,
                attention_mask=bert_sent_mask
            )
        return bert_output[0]

class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads=1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"
        
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, value])
        out = out.reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)
        
        return out

class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=True):
        super(RNNEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.bidirectional_factor = 2 if bidirectional else 1
        
        self.rnn = nn.GRU(
            in_size, hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=bidirectional, 
            batch_first=True
        )
        self.linear = nn.Linear(self.bidirectional_factor * hidden_size, in_size)
        self.output_proj = nn.Linear(self.bidirectional_factor * hidden_size, out_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        output, h_n = self.rnn(x)
        
        if self.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        else:
            h = h_n[-1]
        
        h = self.dropout(h)
        y_1 = self.output_proj(h)
        
        output = self.linear(output)
        output = self.activation(output)
        
        return y_1, output

class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout=0.2, bidirectional=True):
        super(SubNet, self).__init__()
        if dropout == 0.0:
            dropout = 0.0
            
        self.rnn = nn.GRU(
            in_size, hidden_size,
            num_layers=1,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        _, final_states = self.rnn(x)
        h = torch.cat((final_states[0], final_states[1]), -1)
        dropped = self.dropout(h)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        return y_2

class ExpressionSelfAttention(nn.Module):
    def __init__(self, in_size, dropout):
        super(ExpressionSelfAttention, self).__init__()
        self.linear_1 = nn.Linear(in_size, 1)
        self.linear_2 = nn.Linear(2 * in_size, 1)
        self.linear_3 = nn.Linear(2 * in_size, in_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        dropped_1 = self.dropout(x)
        att = torch.sigmoid(self.linear_1(dropped_1))
        vm = torch.mul(att, x).mean(dim=0, keepdim=True)
        vm = vm.repeat(x.shape[0], 1)
        
        vs = torch.cat([x, vm], dim=-1)
        dropped_2 = self.dropout(vs)
        att_new = torch.sigmoid(self.linear_2(dropped_2))
        
        y = torch.mul(att * att_new, vs)
        y_1 = F.relu(self.linear_3(y))
        return y_1

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    
    def __init__(self, x_hidden_size, y_hidden_size, target_hidden_size, dropout, kan_attention_dim=50):
        super().__init__()
        self.x_hidden_size = x_hidden_size
        self.y_hidden_size = y_hidden_size
        self.target_hidden_size = target_hidden_size
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.x_kan_attention = KAN([x_hidden_size * 2, kan_attention_dim, 1])
        self.y_kan_attention = KAN([y_hidden_size * 2, kan_attention_dim, 1])
        self.t_kan_attention = KAN([target_hidden_size * 2, kan_attention_dim, 1])
        

        total_hidden = x_hidden_size + y_hidden_size + target_hidden_size
        self.modal_fusion_kan = KAN([total_hidden, kan_attention_dim, 3])
        
        self.linear_1 = KAN([target_hidden_size, target_hidden_size])
        self.linear_2 = KAN([target_hidden_size, target_hidden_size])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y, target):

        x_att_matrix = self._compute_kan_attention_matrix(x, x, self.x_kan_attention, self.x_hidden_size)
        y_att_matrix = self._compute_kan_attention_matrix(y, y, self.y_kan_attention, self.y_hidden_size)
        t_att_matrix = self._compute_kan_attention_matrix(target, target, self.t_kan_attention, self.target_hidden_size)
        

        fusion_weights = self._compute_dynamic_weights(x, y, target)
        w_x, w_y, w_t = fusion_weights.chunk(3, dim=-1)
        

        fusion_att = (w_x.unsqueeze(-1).unsqueeze(-1) * x_att_matrix + 
                     w_y.unsqueeze(-1).unsqueeze(-1) * y_att_matrix + 
                     w_t.unsqueeze(-1).unsqueeze(-1) * t_att_matrix) + self.bias
        
        fusion_att = F.softmax(fusion_att, dim=-1)
        

        target_att = torch.matmul(fusion_att, target)
        

        dropped = self.dropout(target_att)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        
        return y_2.squeeze(0)
    
    def _compute_kan_attention_matrix(self, query, key, kan_network, hidden_size):
        seq_len, hidden = query.shape
        
        query_expanded = query.unsqueeze(1)  # [L, 1, H]
        key_expanded = key.unsqueeze(0)      # [1, L, H]
        
        query_repeated = query_expanded.repeat(1, seq_len, 1)  # [L, L, H]
        key_repeated = key_expanded.repeat(seq_len, 1, 1)      # [L, L, H]
        
        query_key_pairs = torch.cat([query_repeated, key_repeated], dim=-1)  # [L, L, 2H]
        
        attention_scores = kan_network(query_key_pairs)  # [L, L, 1]
        attention_scores = attention_scores.squeeze(-1)  # [L, L]

        return attention_scores
    
    def _compute_dynamic_weights(self, x, y, target):

        x_global = x.mean(dim=0, keepdim=True)  # [1, H_x]
        y_global = y.mean(dim=0, keepdim=True)  # [1, H_y]
        target_global = target.mean(dim=0, keepdim=True)  # [1, H_t]
        
        modal_features = torch.cat([x_global, y_global, target_global], dim=-1)  # [1, H_x + H_y + H_t]
        
        weights = F.softmax(self.modal_fusion_kan(modal_features), dim=-1)  # [1, 3]
        weights = weights.squeeze(0)  # [3]
        
        return weights
    
class FusionSubNet(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super(FusionSubNet, self).__init__()
        self.rnn = nn.GRU(in_size, in_size, num_layers=1, dropout=0, bidirectional=True, batch_first=True)
        self.linear_1 = KAN([2 * in_size, in_size])
        self.linear_2 = KAN([in_size, in_size])
        
    def forward(self, h, p):
        output, _ = self.rnn(h)
        a_1 = F.relu(self.linear_1(output))
        a_2 = torch.sigmoid(self.linear_2(a_1))
        y = torch.matmul(a_2.permute(1, 0), p).squeeze()
        return y, a_1

def get_mask_from_sequence(sequence, dim):

    return torch.sum(torch.abs(sequence), dim=dim) == 0

class MGMIN_FSA(nn.Module):
    def __init__(self, feature_dims, hidden_dims, dropouts, post_dropouts, post_dim, 
                 train_mode, num_classes, model_config=None):
        super(MGMIN_FSA, self).__init__()
        

        self.text_in = feature_dims["T"]
        self.audio_in = feature_dims["A"]
        self.video_in = feature_dims["V"]
        
        self.text_hidden = hidden_dims["T"]
        self.audio_hidden = hidden_dims["A"]
        self.video_hidden = hidden_dims["V"]
        self.M_in = hidden_dims["M"]
        

        self.text_prob = dropouts["T"]
        self.audio_prob = dropouts["A"]
        self.video_prob = dropouts["V"]
        
        self.post_text_prob = post_dropouts["T"]
        self.post_audio_prob = post_dropouts["A"]
        self.post_video_prob = post_dropouts["V"]
        self.post_fusion_prob = post_dropouts["M"]
        

        self.post_fusion_dim = post_dim["M"]
        self.post_text_dim = post_dim["T"]
        self.post_audio_dim = post_dim["A"]
        self.post_video_dim = post_dim["V"]
        

        self.output_dim = num_classes if train_mode == "classification" else 1
        self.train_mode = train_mode
        

        self.model_config = model_config or {}
        self.use_token_type = self.model_config.get("use_token_type", True)
        self.use_kan_attention = self.model_config.get("use_kan_attention", False)
        self.use_avg_padding = self.model_config.get("use_avg_padding", False)
        self.use_enhanced_same = self.model_config.get("use_enhanced_same", True)
        

        model_type = self.model_config.get("model_type", "chinese_bert")
        model_path = self.model_config.get("model_path", None)
        self.text_enc = LanguageEmbeddingLayer(model_type, model_path)
        

        self.audio_enc = RNNEncoder(self.audio_in, self.audio_hidden, self.audio_in, dropout=self.audio_prob)
        self.video_enc = RNNEncoder(self.video_in, self.video_hidden, self.video_in, dropout=self.video_prob)
        

        self.SA_audio = SelfAttention(self.audio_in, heads=1)
        self.SA_video = SelfAttention(self.video_in, heads=1)
        

        self.audio_same_enc = RNNEncoder(self.audio_in, self.audio_hidden, self.audio_in, dropout=self.audio_prob)
        self.video_same_enc = RNNEncoder(self.video_in, self.video_hidden, self.video_in, dropout=self.video_prob)
        

        if self.use_enhanced_same:
            self.dim_v_same = nn.Sequential(
                nn.BatchNorm1d(self.video_in),
                nn.Linear(self.video_in, self.video_in // 2),
                nn.ReLU(),
                nn.Linear(self.video_in // 2, self.M_in)
            )
            self.dim_a_same = nn.Sequential(
                nn.BatchNorm1d(self.audio_in),
                nn.Linear(self.audio_in, self.audio_in // 2),
                nn.ReLU(),
                nn.Linear(self.audio_in // 2, self.M_in)
            )
            self.dim_t_same = nn.Sequential(
                nn.BatchNorm1d(self.text_in),
                nn.Linear(self.text_in, self.text_in // 2),
                nn.ReLU(),
                nn.Linear(self.text_in // 2, self.M_in)
            )
        else:
            self.bn_v = nn.BatchNorm1d(self.video_in)
            self.bn_a = nn.BatchNorm1d(self.audio_in)
            self.bn_t = nn.BatchNorm1d(self.text_in)
            self.dim_v_same = nn.Linear(self.video_in, self.M_in)
            self.dim_a_same = nn.Linear(self.audio_in, self.M_in)
            self.dim_t_same = nn.Linear(self.text_in, self.M_in)
        

        self.audio_subnet = SubNet(self.audio_in, self.M_in, dropout=self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.M_in, dropout=self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.M_in, dropout=self.text_prob)
        

        self.video_attnet = ExpressionSelfAttention(self.video_in, self.video_prob)
        self.audio_attnet = ExpressionSelfAttention(self.audio_in, self.audio_prob)
        

        self.audio_linear = KAN([self.audio_in, self.audio_hidden])
        self.video_linear = KAN([self.video_in, self.video_hidden])
        self.text_linear = KAN([self.text_in, self.text_hidden])
        

        self.audio_cutnet = CrossModalAttention(
            x_hidden_size=self.video_hidden,  
            y_hidden_size=self.text_hidden,   
            target_hidden_size=self.audio_hidden, 
            dropout=self.audio_prob,

        )

        self.video_cutnet = CrossModalAttention(
            x_hidden_size=self.audio_hidden, 
            y_hidden_size=self.text_hidden,   
            target_hidden_size=self.video_hidden,  
            dropout=self.video_prob,
        )
        
        self.text_cutnet = CrossModalAttention(
            x_hidden_size=self.audio_hidden, 
            y_hidden_size=self.video_hidden,
            target_hidden_size=self.text_hidden, 
            dropout=self.text_prob,
        )
        

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        fusion_input_dim = self.text_hidden + self.video_hidden + self.audio_hidden
        self.post_fusion_layer_1 = nn.Linear(fusion_input_dim, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_dim)
        
        self.fusion_subnet = FusionSubNet(self.post_fusion_dim, self.output_dim, self.post_fusion_prob)
        

        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)
        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        
        if self.use_enhanced_same:
            self.kan_text_dropout = KAN([self.M_in, self.post_text_dim, self.output_dim])
            self.kan_audio_dropout = KAN([self.M_in, self.post_audio_dim, self.output_dim])
            self.kan_video_dropout = KAN([self.M_in, self.post_video_dim, self.output_dim])
        else:
            self.kan_text_dropout = KAN([self.M_in * 2, self.post_text_dim, self.output_dim])
            self.kan_audio_dropout = KAN([self.M_in * 2, self.post_audio_dim, self.output_dim])
            self.kan_video_dropout = KAN([self.M_in * 2, self.post_video_dim, self.output_dim])
        

        self.fusion_layer_1 = nn.Linear(self.M_in * 4, self.M_in)
        self.fusion_layer_2 = nn.Linear(self.M_in, self.output_dim)
        
    def forward(self, input_ids, att_mask, audio_cutx, video_cutx, fs=True, token_type_ids=None):
        if self.use_token_type and token_type_ids is not None:
            text_cutx = self.text_enc(input_ids, att_mask, token_type_ids)
        else:
            text_cutx = self.text_enc(input_ids, att_mask)
        
        text = text_cutx[:, 0, :] 

        text_cutx, audio_cutx, video_cutx = self._align_with_const_padding(
            text_cutx, audio_cutx, video_cutx, fs
        )

        _, audio_cutx = self.audio_enc(audio_cutx)
        _, video_cutx = self.video_enc(video_cutx)
        
        video, _ = self.video_same_enc(video_cutx)
        audio, _ = self.audio_same_enc(audio_cutx)
        
        if self.use_enhanced_same:
            same_v = self.dim_v_same(video)
            same_a = self.dim_a_same(audio)
            same_t = self.dim_t_same(text)
        else:
            same_v = self.dim_v_same(self.bn_v(video))
            same_a = self.dim_a_same(self.bn_a(audio))
            same_t = self.dim_t_same(self.bn_t(text))

        audio_batchs, video_batchs, text_batchs, fusion_batchs = [], [], [], []
        
        for i in range(len(text_cutx)):

            audio_hi = self.audio_subnet(audio_cutx[i])
            video_hi = self.video_subnet(video_cutx[i])
            text_hi = self.text_subnet(text_cutx[i])
            
            audio_batchs.append(audio_hi.unsqueeze(0))
            video_batchs.append(video_hi.unsqueeze(0))
            text_batchs.append(text_hi.unsqueeze(0))
            

            audio_feat = self.audio_linear(audio_cutx[i])
            video_feat = self.video_linear(video_cutx[i])
            text_feat = self.text_linear(text_cutx[i])


            audio_cuth = self.audio_cutnet(video_feat, text_feat, audio_feat)
            video_cuth = self.video_cutnet(audio_feat, text_feat, video_feat)
            text_cuth = self.text_cutnet(audio_feat, video_feat, text_feat)
            

            fusion_h = torch.cat([audio_cuth, video_cuth, text_cuth], dim=-1)
            dropped = self.post_fusion_dropout(fusion_h)
            x_1 = F.relu(self.post_fusion_layer_1(dropped))
            x_2 = F.relu(self.post_fusion_layer_2(x_1))
            _fusion = self.post_fusion_layer_3(x_2)
            
            fusion, M_hi = self.fusion_subnet(x_1, _fusion)
            fusion_batchs.append(fusion.unsqueeze(0))
        

        audio_h = torch.cat(audio_batchs, dim=0)
        video_h = torch.cat(video_batchs, dim=0)
        text_h = torch.cat(text_batchs, dim=0)
        fusions = torch.cat(fusion_batchs, dim=0)
        

        audio_mean = same_a.mean(dim=0, keepdim=True)
        video_mean = same_v.mean(dim=0, keepdim=True)
        text_mean = same_t.mean(dim=0, keepdim=True)
        

        fusion_input = torch.cat([fusions, same_t, same_a, same_v], dim=-1)
        f1 = self.fusion_layer_1(fusion_input)
        output_fusion = self.fusion_layer_2(f1)
        

        if self.use_enhanced_same:
            x_t = self.post_text_dropout(text_h)
            x_a = self.post_audio_dropout(audio_h)
            x_v = self.post_video_dropout(video_h)
        else:
            x_t = self.post_text_dropout(torch.cat([text_h, same_t], dim=-1))
            x_a = self.post_audio_dropout(torch.cat([audio_h, same_a], dim=-1))
            x_v = self.post_video_dropout(torch.cat([video_h, same_v], dim=-1))
        
        output_text = self.kan_text_dropout(x_t)
        output_audio = self.kan_audio_dropout(x_a)
        output_video = self.kan_video_dropout(x_v)
        

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            
            "T_same": same_t,
            "A_same": same_a,
            "V_same": same_v,
            
            "pre_t": output_text,
            "pre_a": output_audio,
            "pre_v": output_video,
            
            'M': output_fusion,
            'T': text_mean,
            'A': audio_mean,
            'V': video_mean,
        }
        
        return res
    
    def _align_with_const_padding(self, text_cutx, audio_cutx, video_cutx, fs):
        if fs:
            l_av = max(audio_cutx.shape[1], video_cutx.shape[1])
            length_padded = text_cutx.shape[1]
            pad_before = int((length_padded - l_av)/2)
            pad_after = length_padded - l_av - pad_before
            audio_cutx = F.pad(audio_cutx, (0, 0, pad_before, pad_after, 0, 0), "constant", 1e-6)
            video_cutx = F.pad(video_cutx, (0, 0, pad_before, pad_after, 0, 0), "constant", 1e-6)
        else:
            l_t = text_cutx.shape[1]
            l_v = video_cutx.shape[1]
            length_padded = audio_cutx.shape[1]
            pad_before_t = int((length_padded - l_t)/2)
            pad_before_v = int((length_padded - l_v)/2)
            pad_after_t = length_padded - l_t - pad_before_t
            pad_after_v = length_padded - l_v - pad_before_v
            text_cutx = F.pad(text_cutx, (0, 0, pad_before_t, pad_after_t, 0, 0), "constant", 1e-6)
            video_cutx = F.pad(video_cutx, (0, 0, pad_before_v, pad_after_v, 0, 0), "constant", 1e-6)
        
        return text_cutx, audio_cutx, video_cutx
    
