from torch.utils.data import Dataset
import torch
import numpy as np

from data_loader.rbmixin import RangeBasedMixin

def default_collate_fn(items, experts_info):
    all_caption = []
    all_caption_t = []
    all_features = {}
    all_features_t = {}
    all_features_mask = {}

    bs = len(items)

    for mod_name, einfo in experts_info.items():
        max_tok = einfo["max_tok"]
        dim = einfo["dim"]
        all_features[mod_name] = torch.zeros(bs, max_tok, dim)
        all_features_t[mod_name] = torch.zeros(bs, max_tok)
        all_features_mask[mod_name] = torch.zeros(bs, max_tok)

    for batch_idx, (caption, caption_t, features, features_t) in enumerate(items):
        all_caption.append(caption)
        all_caption_t.append(caption_t)
        for mod_name in features.keys():
            max_tok = experts_info[mod_name]["max_tok"]
            mod_feat = features[mod_name]
            mod_feat_t = np.array(features_t[mod_name])
            assert len(mod_feat) == len(mod_feat_t), (len(mod_feat), len(mod_feat_t))
            if np.isnan(mod_feat_t.sum()):
                mod_feat_t = np.zeros(len(mod_feat_t))
                mod_feat_t[:] = 1
            else:
                mod_feat_t = mod_feat_t - mod_feat_t[:,0].min()
                mod_feat_t = 2 + (mod_feat_t[:,1] + mod_feat_t[:,0]) / 2 # (ntok,)
            all_features[mod_name][batch_idx,:len(mod_feat)] = torch.from_numpy(mod_feat[:max_tok].copy())
            all_features_t[mod_name][batch_idx,:len(mod_feat)] = torch.from_numpy(mod_feat_t[:max_tok].copy())
            all_features_mask[mod_name][batch_idx, :len(mod_feat)] = 1

    all_caption_t = np.array(all_caption_t)
    return all_caption, all_caption_t, all_features, all_features_t, all_features_mask


class BaseDataset(RangeBasedMixin, Dataset):
    def __init__(self, **kwargs):
        self.data_dir = kwargs.pop('data_dir')
        self.cut_name = kwargs.pop('cut_name')
        self.split_name = kwargs.pop('split_name')
        self.experts_info = kwargs.pop('experts_info')
        self.training = kwargs.pop('training')
        self.restrict_test_captions = None

        self.configure_train_test_splits(self.cut_name, self.split_name)
        super().__init__(**kwargs)
    
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx, capidx=-1):
        idx = idx % len(self.vid_list)
        vid = self.vid_list[idx]
        if self.restrict_test_captions is not None and vid in self.restrict_test_captions:
            capidx = self.restrict_test_captions[vid]
        sample_data = self.get_sample_data(vid, capidx=capidx)
        captions = sample_data["captions"]
        captions_t = sample_data["captions_t"]
        features = sample_data["features"]
        features_t = sample_data["features_t"]


        assert len(captions) == 1, len(captions)
        caption = captions[0]
        caption_t = captions_t[0]
        return caption, caption_t, features, features_t

    def collate_fn(self, items):
        return default_collate_fn(items, self.experts_info)

    def get_val_captions(self, mode='all', seed=None):
        if mode == 'all':
            capidx = None
        elif mode == 'rand_1':
            capidx = -1
        else:
            raise NotImplementedError(f'Unknown mode={mode}')
        all_captions = []
        for vid in self.vid_list:
            sample_data = self.get_sample_data(vid, capidx=capidx, caponly=True, seed=seed)
            captions = sample_data["captions"]
            captions_t = sample_data["captions_t"]
            captions_idxs = sample_data["captions_idxs"]
            assert len(captions) == len(captions_t) == len(captions_idxs), (len(captions), len(captions_t), len(captions_idxs))

            for capidx1, cap, (t0, t1) in zip(captions_idxs, captions, captions_t):
                all_captions.append((vid, capidx1, cap, t0, t1))
        return all_captions

class RandCapMixin:
    def get_val_captions(self, mode='rand_1', seed=42):
        return super().get_val_captions(mode, seed)
