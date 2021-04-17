import sys
from collections import OrderedDict
import types
from models.mmt.bert_mmt import BertModel

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertModel as TxtBertModel
from transformers import AutoTokenizer
import torch
import os


def state_dict_txt_bert(state):
    state_dict1 = {}
    state_dict = state
    for key in state_dict.keys():
        if key.startswith('text_GU.'):
            key1 = key.replace('text_GU.', 'text_gu.')
        else:
            key1 = key
        if not (\
                key.startswith('txt_bert.') or \
                key.startswith('text_gu.') or \
                key.startswith('text_GU.') or \
                'moe_fc_txt' in key or \
                'moe_txt_dropout' in key \
        ):
            continue
        state_dict1[key1] = state_dict[key]
    return state_dict1

def state_dict_vid_bert(state):
    state1 = {}
    state_dict = state
    for key in state_dict:
        if key.startswith('vid_bert') or key.startswith('video_dim_reduce') or key.startswith('video_dim_reduce'):
            state1[key] = state_dict[key]
    return state1

class ContextGating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn, normalize):
        super(GatedEmbeddingUnit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)
        self.normalize = normalize

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


class ReduceDim(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        return x

def get_maxp(embd, mask):
    # (bs, ntok, embdim)
    # (bs, ntok) 1==token, 0==pad
    mask = mask.clone().to(dtype=torch.float32)
    all_pad_idxs = (mask==0).all(dim=1)
    mask[mask == 0] = float('-inf')
    mask[mask == 1] = 0
    maxp = (embd + mask[..., None]).max(dim=1)[0] # (bs, embdim)
    maxp[all_pad_idxs] = 0 # if there is not embeddings, use zeros
    return maxp

def pad(x, max_length):
    bs, n = x.shape
    if n < max_length:
        padding = torch.zeros(bs, max_length - n, dtype=x.dtype)
        x = torch.cat([x, padding], dim=1)
    return x


class BertTXT(nn.Module):
    def __init__(self,
                 txt_bert_config='/ssd/ssd_srv79/models/huggingface/bert-base-cased',
                 max_length=30,
                 modalities=['tf_s3dg_k600'],
                 add_special_tokens=True,
                 add_dot=True,
                 same_dim=512,
                 txt_bert_params = {
                    'hidden_dropout_prob': 0.1,
                    'attention_probs_dropout_prob': 0.1,
                 },
    ):
        super().__init__()
        self.orig_mmt_comaptible = int(os.environ.get('ORIG_MMT_COMPAT', 0))
        if self.orig_mmt_comaptible:
            print('ORIG_MMT_COMPAT')
        dout_prob = txt_bert_params['hidden_dropout_prob']
        self.add_dot = add_dot
        self.add_special_tokens = add_special_tokens
        self.modalities = modalities
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(txt_bert_config)
        self.txt_bert = TxtBertModel.from_pretrained(txt_bert_config, **txt_bert_params)
        text_dim = self.txt_bert.config.hidden_size
        
        self.text_gu = nn.ModuleDict()
        for mod in self.modalities:
            self.text_gu[mod] = GatedEmbeddingUnit(text_dim,
                                                   same_dim,
                                                   use_bn=True,
                                                   normalize=True)
        
        self.moe_fc_txt = nn.ModuleDict()
        self.moe_txt_dropout = nn.Dropout(dout_prob)
        for mod in self.modalities:
            self.moe_fc_txt[mod] = nn.Linear(text_dim, 1)
            
    @property
    def device(self):
        return next(self.parameters()).data.device

    def compute_weights_from_emb(self, embd):
        embd = self.moe_txt_dropout(embd)
        m = len(self.modalities)
        moe_weights = th.cat([self.moe_fc_txt[mod](embd) for mod in self.modalities], dim=-1)
        moe_weights = F.softmax(moe_weights, dim=1)
        return moe_weights
    
    def forward(self, text_list):
        if self.add_dot:
            text_list1 = []
            for x in text_list:
                x = x.strip()
                if x[-1] not in ('.', '?', '!'):
                    x = x + '.'
                text_list1.append(x)
            text_list = text_list1
        device = self.device
        encoded_inputs = self.tokenizer(text_list,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=self.add_special_tokens,
                padding=True,
                return_tensors='pt')
        bs, ntok = encoded_inputs['input_ids'].shape
        if self.orig_mmt_comaptible:
            encoded_inputs = {key: pad(value, self.max_length).to(device) for key, value in encoded_inputs.items()}
            encoded_inputs['head_mask'] = None
        else:
            encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
        x = self.txt_bert(**encoded_inputs)[0] # (bs, max_tokens, hidden_size)
        # authors of MMT take token 0 and think that it is CLS
        # but they dont provide CLS token to input 
        text = x[:,0,:]
        text_embd = []
        for mod in self.modalities:
            layer = self.text_gu[mod] # this layer containg F.normalize
            text_ = layer(text) # (bs, d_model) this is unit-length embs
            if self.orig_mmt_comaptible:
                text_ = F.normalize(text_)
            text_ = text_.unsqueeze(1) # (bs, 1, d_model)
            text_embd.append(text_)
        text_embd = torch.cat(text_embd, dim=1) # (bs, nmods, d_model)
        if len(self.modalities) > 1:
            text_weights = self.compute_weights_from_emb(text) # (bs, nmods)
            embd_wg = text_embd * text_weights[..., None] # (bs, nmods, d_model)
        else:
            embd_wg = text_embd
        
        bs = embd_wg.size(0)
        text_embd = embd_wg.view(bs, -1)
        
        return text_embd

class BertVID(nn.Module):
    def __init__(self,
                 expert_dims={
                     'tf_s3dg_k600': dict(idx=0, dim=1024),
                 },
                 same_dim=512,
                 vid_bert_params=OrderedDict([
                    ('vocab_size_or_config_json_file', 10),
                    ('hidden_size', 512),
                    ('num_hidden_layers', 4),
                    ('num_attention_heads', 4),
                    ('intermediate_size', 3072),
                    ('hidden_act', 'gelu'),
                    ('hidden_dropout_prob', 0.1),
                    ('attention_probs_dropout_prob', 0.1),
                    ('max_position_embeddings', 32),
                    ('type_vocab_size', 19),
                    ('initializer_range', 0.02),
                    ('layer_norm_eps', 1e-12)
                ]),
    ):
        super().__init__()
        self.modalities = list(expert_dims.keys())
        self.same_dim = same_dim
        self.expert_dims = expert_dims
        self.vid_bert_params = vid_bert_params
        self.hidden_size = self.vid_bert_params["hidden_size"]

        vid_bert_config = types.SimpleNamespace(**self.vid_bert_params)
        self.vid_bert = BertModel(vid_bert_config)
        
        self.video_dim_reduce = nn.ModuleDict()
        for mod in self.modalities:
            in_dim = expert_dims[mod]['dim']
            self.video_dim_reduce[mod] = ReduceDim(in_dim, self.hidden_size)
    
        if same_dim != self.hidden_size:
            self.video_dim_reduce_out = nn.ModuleDict()
            for mod in self.modalities:
                self.video_dim_reduce_out[mod] = ReduceDim(self.hidden_size, same_dim)
    
    @property
    def device(self):
        return next(self.parameters()).data.device
    
    def forward(self,
                features, # embs from pretrained models {modality: (bs, ntok, embdim)}
                features_t, # timings {modality: (bs, ntok)} each value is (emb_t_start + emb_t_end) / 2
                features_ind, # mask (modality: (bs, ntok))
                features_maxp=None,
        ):
        device = self.device
        #for mod in features_t.keys():
        #    import pdb; pdb.set_trace()
        #    features_t[mod][features_ind[mod]==0] = 1
        experts_feats = dict(features)
        experts_feats_t = dict(features_t)
        experts_feats_ind = dict(features_ind)
        ind = {} # 1 if there is at least one non-pad token in this modality 
        for mod in self.modalities:
            ind[mod] = th.max(experts_feats_ind[mod], 1)[0]
    
        for mod in self.modalities:
            layer = self.video_dim_reduce[mod]
            experts_feats[mod] = layer(experts_feats[mod])
            
        bs = next(iter(features.values())).size(0)
        ids_size = (bs,)
        input_ids_list = []
        token_type_ids_list = []  # Modality id
        # Position (0 = no position, 1 = unknown, >1 = valid position)
        position_ids_list = []
        features_list = []  # Semantics
        attention_mask_list = []  # Valid token or not

        modality_to_tok_map = OrderedDict()

        # 0=[CLS] 1=[SEP] 2=[AGG] 3=[MAXP] 4=[MNP] 5=[VLAD] 6=[FEA]
        # [CLS] token
        tok_id = 0
        input_ids_list.append(th.full(ids_size, 0, dtype=th.long))
        token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
        position_ids_list.append(th.full(ids_size, 0, dtype=th.long).to(device))
        features_list.append(th.full((bs, self.hidden_size), 0, dtype=th.float).to(device))
        attention_mask_list.append(th.full(ids_size, 1, dtype=th.long).to(device))

        # Number of temporal tokens per modality
        max_expert_tokens = OrderedDict()
        for modality in self.modalities:
            max_expert_tokens[modality] = experts_feats[modality].size()[1]

        # Clamp the position encoding to [0, max_position_embedding - 1]
        max_pos = self.vid_bert_params['max_position_embeddings'] - 1
        for modality in self.modalities:
            experts_feats_t[modality].clamp_(min=0, max=max_pos)
            experts_feats_t[modality] = experts_feats_t[modality].long().to(device)

        for modality in self.modalities:
            token_type = self.expert_dims[modality]['idx']
            tok_id += 1
            
            # add aggregation token
            modality_to_tok_map[modality] = tok_id
            input_ids_list.append(th.full(ids_size, 2, dtype=th.long))
            token_type_ids_list.append(th.full(ids_size, token_type, dtype=th.long))
            position_ids_list.append(th.full(ids_size, 0, dtype=th.long).to(device))
            layer = self.video_dim_reduce[modality]
            if features_maxp is not None:
                feat_maxp = features_maxp[modality]
            else:
                feat_maxp = get_maxp(features[modality], experts_feats_ind[modality]) # (bs, embdim)
            features_list.append(layer(feat_maxp))
            attention_mask_list.append(ind[modality].to(dtype=th.long).to(device))
            
            # add expert tokens
            for frame_id in range(max_expert_tokens[modality]):
                tok_id += 1
                position_ids_list.append(experts_feats_t[modality][:, frame_id])
                input_ids_list.append(th.full(ids_size, 6, dtype=th.long))
                token_type_ids_list.append(th.full(ids_size, token_type, dtype=th.long))
                features_list.append(experts_feats[modality][:, frame_id, :])
                attention_mask_list.append(experts_feats_ind[modality][:, frame_id].to(dtype=th.long))
                
        features = th.stack(features_list, dim=1).to(self.device)
        input_ids = th.stack(input_ids_list, dim=1).to(self.device)
        token_type_ids = th.stack(token_type_ids_list, dim=1).to(self.device)
        position_ids = th.stack(position_ids_list, dim=1).to(self.device)
        attention_mask = th.stack(attention_mask_list, dim=1).to(self.device)
        vid_bert_output = self.vid_bert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        features=features)
        #return input_ids, attention_mask, token_type_ids, position_ids, features
        last_layer = vid_bert_output[0]
        vid_embd = last_layer[:, 0]
        #return vid_embd
        #experts = {}
        #for _, modality in enumerate(self.modalities):
        #    experts[modality] = last_layer[:, modality_to_tok_map[modality]]
        
        experts = []
        for modality in self.modalities:
            emb = last_layer[:, modality_to_tok_map[modality]]
            if self.same_dim != self.hidden_size:
                emb = self.video_dim_reduce_out[mod](emb)
            agg_tok_out = F.normalize(emb, dim=1)
            #if ind[modality].sum() > 0:
            #    import pdb; pdb.set_trace()
            experts.append(agg_tok_out) # (bs, embdim)
        experts = torch.cat(experts, dim=1)
        return experts

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    #sys.path.insert(0, '/home/wx587276/shared_folder/src/mmt_orig')
    from model.model import CENet
    model_pt = '/ssd/ssd_srv79/dza/mmt/exps/cleaning/MSRVTT_jsfusion_trainval/orig/2/trained_model.pth'
    state = torch.load(model_pt, map_location=torch.device('cpu'))
    bs = 2
    
    expert_dims = {
        "face": {
            "dim": 512,
            "idx": 3,
            "max_tok": 30 
        },
        "ocr": {
            "dim": 300,
            "idx": 7,
            "max_tok": 30
        },
        "rgb": {
            "dim": 2048,
            "idx": 5,
            "max_tok": 30
        },
        "s3d": {
            "dim": 1024,
            "idx": 1,
            "max_tok": 30
        },
        "scene": {
            "dim": 2208,
            "idx": 9,
            "max_tok": 30
        },
        "speech": {
            "dim": 300,
            "idx": 6,
            "max_tok": 30
        },
        "vggish": {
            "dim": 128,
            "idx": 2,
            "max_tok": 30
        }
    }

    def rand_batch(expert_dims, bs):
        features = {}
        features_t = {}
        features_ind = {}
        for mod, info in expert_dims.items():
            dim = info['dim']
            ntok = info['max_tok']
            features[mod] = torch.randn(bs, ntok, dim)
            features_ind[mod] = torch.ones(bs, ntok)
            if mod in ['s3d', 'vggish', 'scene']:
                t1 = torch.arange(0, ntok)
                t2 = torch.arange(1, ntok+1)
                tt = 2 + (t1+t2) / 2
                features_t[mod] = tt[None,...].expand(bs,ntok).clone()
            elif mod in ['rgb']:
                t1 = torch.arange(0, ntok, 0.2)[:ntok]
                t2 = torch.arange(0.2, ntok, 0.2)[:ntok]
                tt = 2 + (t1+t2) / 2
                features_t[mod] = tt[None,...].expand(bs,ntok).clone()
            else:
                features_t[mod] = torch.ones(bs, ntok)
        return features, features_t, features_ind

    features, features_t, features_ind = rand_batch(expert_dims, bs=bs)
    features_ind['s3d'] = torch.zeros_like(features_ind['face'])

    features_maxpool = {modality: get_maxp(features[modality], features_ind[modality]) for modality in expert_dims.keys()}
    features_avgpool = features_maxpool

    text_list = ['I love my wife', 'my son is growing too fast']


    vid_bert_params = OrderedDict([
        ('vocab_size_or_config_json_file', 10),
        ('hidden_size', 512),
        ('num_hidden_layers', 4),
        ('num_attention_heads', 4),
        ('intermediate_size', 3072),
        ('hidden_act', 'gelu'),
        ('hidden_dropout_prob', 0.1),
        ('attention_probs_dropout_prob', 0.1),
        ('max_position_embeddings', 32),
        ('type_vocab_size', 19),
        ('initializer_range', 0.02),
        ('layer_norm_eps', 1e-12)
    ])

    txt_bert_params = OrderedDict([
        ('hidden_dropout_prob', 0.1),
        ('attention_probs_dropout_prob', 0.1)
    ])


    print('Loading CENet ...')
    cenet = CENet(
        l2renorm=False,
        expert_dims=expert_dims,
        tokenizer=None,
        keep_missing_modalities=True,
        test_caption_mode='indep',
        freeze_weights=False,
        mimic_ce_dims=False,
        concat_experts=False,
        concat_mix_experts=False,
        use_experts='origfeat',
        txt_inp='bertftn',
        txt_agg='bertftn',
        txt_pro='gbn',
        txt_wgh='emb',
        vid_inp='both',
        vid_cont='bert',
        vid_wgh='none',
        pos_enc='tint',
        out_tok='mxp',
        use_mask='nomask',
        same_dim=512,
        vid_bert_params=vid_bert_params,
        txt_bert_params=txt_bert_params,
        agg_dims=None,
        normalize_experts=True)
    state1 = {key.replace('text_GU', 'text_gu'): val for key, val in state['state_dict'].items()}
    cenet.load_state_dict(state1)
    cenet.eval()

    print('Loading BertVID ...')
    vid_bert = BertVID(expert_dims=expert_dims, vid_bert_params=vid_bert_params)
    vid_bert.eval()
    vid_bert.load_state_dict(state_dict_vid_bert(state['state_dict']), strict=True)

    print('Loading BertTXT ...')
    txt_bert = BertTXT(txt_bert_params=txt_bert_params, modalities=list(expert_dims.keys()))
    txt_bert.eval()
    txt_bert.load_state_dict(state_dict_txt_bert(state['state_dict']), strict=True)

    vid_embd = vid_bert(features=features, features_t=features_t, features_ind=features_ind)
    txt_embd = txt_bert(text_list)

    max_text_words = 30
    ce_txt_input = torch.zeros(bs, 1, max_text_words, 2).long()
    text_list1 = []
    for x in text_list:
        x = x.strip()
        if x[-1] not in ('.', '?', '!'):
            x = x + '.'
        text_list1.append(x)
    text_list = text_list1
    encoded_inputs = txt_bert.tokenizer(text_list,
            max_length=max_text_words,
            truncation=True,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt')
    toks = encoded_inputs['input_ids']
    bs_, ntoks = toks.shape
    ce_txt_input[:,0,:ntoks,0] = toks
    ce_txt_input[:,0,:ntoks,1] = (toks>0).long()

    ori_output = cenet(
        token_ids=ce_txt_input,
        features=dict(features),
        features_t=dict(features_t),
        features_ind=dict(features_ind),
        features_avgpool=dict(features_avgpool),
        features_maxpool=dict(features_maxpool),
        query_masks=None,
        out='embds'
    )

    wgh_ori_text_embds = (ori_output['text_embds'] * ori_output['text_weights'].transpose(1,2)[..., None]).reshape(bs, -1) 
    print('TEXT:', (wgh_ori_text_embds - txt_embd).norm(dim=1).max().item(), wgh_ori_text_embds.norm())
    print('VIDEO:', (ori_output['vid_embds'].reshape(bs, -1) - vid_embd).norm(dim=1).max().item(), vid_embd.norm())


