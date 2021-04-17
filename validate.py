from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset

import torch
import torch.distributed as dist
import numpy as np


def move_dict_to_device(res, device, only_tensors=True):
  """Move a dictionnary to another device memory."""
  for key in list(res.keys()):
    value = res[key]
    if isinstance(value, np.ndarray):
      res[key] = torch.from_numpy(res[key])
      if device is not None:
        res[key] = res[key].to(device)
    elif isinstance(value, torch.Tensor):
      if device is not None:
        res[key] = value.to(device)
    elif isinstance(value, collections.OrderedDict) or isinstance(value, dict):
      res[key] = move_dict_to_device(res[key], device)
    else:
      if only_tensors:
        res.pop(key)
  return res

class DummySampler(torch.utils.data.Sampler):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)

class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, key):
        item_idx, cap_idx = key
        return self.dataset.__getitem__(item_idx, cap_idx)

def _batchify(batch_size, iterable):
    items = []
    for item in iterable:
        items.append(item)
        if len(items) == batch_size:
            yield items
            items = []
    if len(items) > 0:
        yield items

def batchify(batch_size, *iterables):
    b_iterables = [_batchify(batch_size, it) for it in iterables]
    while True:
        try:
            yield tuple([next(b_it) for b_it in b_iterables])
        except StopIteration:
            return

@torch.no_grad()
def validate(video_model, text_model, dataset, embdim, rank, world_size, video_batch_size, text_batch_size, with_v2t=True):
    device = torch.device('cuda')
    vidxs = list(range(rank, len(dataset), world_size))

    seed = 42
    captions = dataset.get_val_captions(seed=seed) # [(vid, capidx, text, t0, t1), ...]

    # capsegmidx --> (temb_idx, vemb_idx)

    capsegmidx_2_embidxs = {}
    temb_idx_cnt = 0
    vemb_idx_cnt = 0
    text_2_idx = OrderedDict()
    segm_2_idx = OrderedDict()
    segm_2_vidcap0 = {}
    cap_2_capsegmidx = {}
    for capsegmidx, (vid, capidx, cap, t0, t1) in enumerate(captions):
        if cap not in cap_2_capsegmidx:
            cap_2_capsegmidx[cap] = [capsegmidx]
        else:
            cap_2_capsegmidx[cap].append(capsegmidx)
        if cap in text_2_idx:
            temb_idx = text_2_idx[cap]
        else:
            temb_idx = temb_idx_cnt
            temb_idx_cnt += 1
            text_2_idx[cap] = temb_idx

        segm = (vid, t0, t1)
        if segm in segm_2_idx:
            vemb_idx = segm_2_idx[segm]
        else:
            vemb_idx = vemb_idx_cnt
            vemb_idx_cnt += 1
            segm_2_idx[segm] = vemb_idx
            segm_2_vidcap0[segm] = (vid, capidx)
        capsegmidx_2_embidxs[capsegmidx] = (temb_idx, vemb_idx)
    #from remote_pdb import set_trace; set_trace()
    # dumps texts
    texts = list(text_2_idx.keys())
    embs_txt = torch.zeros(len(texts), embdim).cuda()
    local_text_idxs = range(rank, len(texts), world_size)
    local_texts = [texts[idx] for idx in local_text_idxs]
    for batch_texts, batch_idxs in batchify(text_batch_size, local_texts, local_text_idxs):
        embs = text_model(batch_texts)
        embs_txt[batch_idxs] = embs
    if world_size > 1:
        dist.all_reduce(embs_txt)

    #dump video segments
    segms = list(segm_2_idx.keys())
    embs_vid = torch.zeros(len(segms), embdim).cuda()
    local_segm_idxs = range(rank, len(segms), world_size)
    local_segms = [segms[idx] for idx in local_segm_idxs]
    # now we gonna convert local_segms to pairs (idx, capidx)
    vid_2_itemidx = {vid: itemidx for itemidx, vid in enumerate(dataset.vid_list)}
    access_keys = []
    for segm in local_segms:
        vid, capidx = segm_2_vidcap0[segm]
        itemidx = vid_2_itemidx[vid]
        access_keys.append((itemidx, capidx))
    embs_vid = torch.zeros(len(segms), embdim).cuda()

    loader = torch.utils.data.DataLoader(
            dataset=DatasetWrapper(dataset),
            batch_size=video_batch_size,
            sampler=DummySampler(access_keys),
            num_workers=0,
            collate_fn=getattr(dataset, 'collate_fn', None))
    segms_it = iter(local_segms)
    for batch in loader:
        if len(batch) == 6:
            _captions, _captions_t, features, features_t, features_mask, features_maxp = batch
        elif len(batch) == 5:
            _captions, _captions_t, features, features_t, features_mask = batch
            features_maxp = None
        bs = next(iter(features.values())).size(0)
        vidxs = [segm_2_idx[next(segms_it)] for _ in range(bs)]
        features = move_dict_to_device(features, device)
        features_t = move_dict_to_device(features_t, device)
        features_mask = move_dict_to_device(features_mask, device)
        if features_maxp is not None:
            features_maxp = move_dict_to_device(features_maxp, device)
        embs = video_model(features, features_t, features_mask, features_maxp)
        embs_vid[vidxs] = embs
    if world_size > 1:
        dist.all_reduce(embs_vid)
    assert not embs_vid.isnan().any()

    all_topk = torch.zeros(len(captions)).cuda()
    if with_v2t:
        all_topk_v2t = torch.zeros(len(captions)).cuda()
        same_segments = defaultdict(list)
        for x_idx, (x_vid, _, _, x_t0, x_t1) in enumerate(captions):
            temb_idx, vemb_idx = capsegmidx_2_embidxs[x_idx]
            same_segments[(x_vid, x_t0, x_t1)].append(temb_idx)
    else:
        all_topk_v2t = None
    for capsegmidx in range(rank, len(captions), world_size):
        vid, _capidx, cap, t0, t1 = captions[capsegmidx]
        samecap_idxs = cap_2_capsegmidx[cap] # this is capsegmidxs
        ign_vid_idxs = [capsegmidx_2_embidxs[scidx][1] for scidx in samecap_idxs]
        (temb_idx, vemb_idx) = capsegmidx_2_embidxs[capsegmidx]
        temb = embs_txt[temb_idx]
        scores = torch.matmul(embs_vid, temb)
        score = scores[vemb_idx].clone()
        assert vemb_idx in ign_vid_idxs
        scores[ign_vid_idxs] = -999
        best = scores >= score
        all_topk[capsegmidx] = best.sum() # how many segments have same score or higher

        if with_v2t:
            vemb = embs_vid[vemb_idx]
            scores = torch.matmul(embs_txt, vemb)
            ign_idxs = same_segments[(vid, t0, t1)]
            scores[ign_idxs] = -999 # ignore other captions which describing this segment
            best = scores >= score
            all_topk_v2t[capsegmidx] = best.sum()

    if world_size > 1:
        dist.all_reduce(all_topk)
        if with_v2t:
            dist.all_reduce(all_topk_v2t)
        dist.barrier()

    return all_topk, all_topk_v2t


