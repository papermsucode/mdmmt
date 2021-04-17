import os
import json
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import re
import subprocess
import argparse

def find_segment_t0(arr, t):
    if arr[0][0] >= t:
        return 0
    prev_idx = 0
    for idx, s in enumerate(arr):
        if s[0] <= t and t <= s[1]:
            return idx
        elif t < s[0]:
            return prev_idx
        prev_idx = idx
    assert False, (arr, t)

def find_segment_t1(arr, t):
    if arr[-1][1] <= t:
        return len(arr) - 1
    prev_idx = None
    
    for idx, s in enumerate(arr):
        if s[0] <= t and t <= s[1]:
            return idx
        elif t < s[0]:
            return idx
        prev_idx = idx
    assert False, (arr, t)

class IdxIter:
    def __init__(self, fname_idx, emb_bsz, skip_pred=None):
        self.fname_idx = fname_idx
        self.emb_bsz = emb_bsz
        self.prev_video_path = None
        self.skip_pred = skip_pred

    def __iter__(self):
        offt_idx = 0
        offt_emb = 0
        with open(self.fname_idx) as f:
            for line in f.readlines():
                if line.startswith('VIDEO\t'):
                    offt_idx += len(line)
                    if self.prev_video_path is not None and skip_flag == False:
                        yield self.prev_video_path, self.prev_offt_idx, self.prev_offt_emb, np.array(timings), timings_offt
                    timings = []
                    timings_offt = []
                    video_path = line.strip().split('\t')[1]
                    if self.skip_pred:
                        skip_flag = self.skip_pred(video_path)
                    else:
                        skip_flag = False
                        
                    self.prev_video_path = video_path
                    self.prev_offt_idx = offt_idx
                    self.prev_offt_emb = offt_emb
                    
                else:
                    if skip_flag == False:
                        timings_offt.append(offt_idx)
                        a, b = line.strip().split('\t')
                        timings.append((float(a), float(b)))
                    offt_idx += len(line)
                    offt_emb += self.emb_bsz

            if len(timings) > 0 and skip_flag == False:
                # check case when there is no embeddings
                yield self.prev_video_path, self.prev_offt_idx, self.prev_offt_emb, np.array(timings), timings_offt


def arg_modality(x):
    mod_name, dim, prefix = x.split(":")
    dim = int(dim)
    return mod_name, dim, prefix


def find_shards(prefix):
    shards = []
    root = os.path.dirname(prefix)
    for fname in os.listdir(root):
       fname = os.path.join(root, fname)
       if fname.startswith(prefix) and fname.endswith('.idx'):
          path_idx = os.path.join(root, fname)
          path_emb = path_idx.replace('.idx', '.emb')
          shards.append((path_emb, path_idx))
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', action='append', help='mod_name:dim:prefix', type=arg_modality)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--dataset', required=True, choices=['msrvtt', 'ActivityNet', 'lsmdc_publictest'])
    args = parser.parse_args()

    j = {}
    if args.dataset == 'msrvtt':
        with open('mdata/msrvtt_mdata_test.json') as f:
            jtest = json.load(f)
            j.update(jtest)
    #with open('/ssd/ssd_srv79/datasets/msrvtt/msrvtt_mdata_val.json') as f:
    #    jval = json.load(f)
    #    j.update(jval)
    #with open('/ssd/ssd_srv79/datasets/msrvtt/msrvtt_mdata_train.json') as f:
    #    jtrain = json.load(f)
    #    j.update(jtrain)
    elif args.dataset == 'ActivityNet':
        with open('mdata/mdata_val.json') as f:
            jval = json.load(f)
            j.update(jval)
        #with open('/ssd/ssd_srv79/datasets/ActivityNet/mdata_train.json') as f:
        #    jtrain = json.load(f)
        #    j.update(jtrain)
    elif args.dataset == 'lsmdc_publictest':
        with open('mdata/lsmdc_mdata_test.json') as f:
            jtest = json.loads(f.read())
            j.update(jtest)
    else:
        raise NotImplementedError(args.dataset)


    mod_dim = {}
    mod_names = []
    dump_prefs = []
    for mod_name, dim, prefix in args.modality:
        mod_dim[mod_name] = dim
        mod_names.append(mod_name)
        dump_prefs.append(prefix)
    out_root = os.path.join(args.output_root, 'capts')
    os.makedirs(out_root, exist_ok=True)

    index = dict() # vid --> {modK --> offt_idx, offt_emb, timings, timings_offt, ...}
    for mod_name, prefix in zip(mod_names, dump_prefs):
        shards = find_shards(prefix)
        assert len(shards) > 0, prefix
        for emb_fname, idx_fname in tqdm(shards):
            embdim = mod_dim[mod_name]
            for vid_path, offt_idx, offt_emb, timings, timings_offt in iter(IdxIter(idx_fname, embdim*4, skip_pred=None)):
                if args.dataset in ['msrvtt', 'ActivityNet', 'lsmdc_publictest']:
                    vid = os.path.splitext(os.path.basename(vid_path))[0]
                else:
                    raise NotImplementedError(f'unknown dataset {args.dataset}')
                if vid not in index:
                    index[vid] = {}
                index[vid][mod_name] = (emb_fname, idx_fname, offt_emb, offt_idx, timings, timings_offt)
    print('Len index', len(index))
    K = len( set(index.keys()) & set(j.keys()) )
    #import pdb; pdb.set_trace()
    print()
    print('Len intersection index and mdata', K)
    assert K > 0, args.dataset

    skept = []
    processed = set()
    all_mdata = {}
    verbose = False
    for vid_idx, (vid, mdata) in enumerate(tqdm(j.items())):
        if vid not in index:
            skept.append(vid)
            if verbose:
                print(f'SKIP[{len(skept)} | {vid_idx}]. not in index', vid)
            continue
        I = index[vid]
        
        fnames_l2 = []
        dims = []
        offts_l3 = []
        duration = 0
        for mod_name in mod_names:
            if mod_name in I:
                emb_fname, idx_fname, offt_emb, offt_idx, timings, timings_offt = I[mod_name]
            else:
                emb_fname, idx_fname, offt_emb, offt_idx, timings, timings_offt = '', '', -1, -1, [], []
            fnames_l2.append(emb_fname)
            fnames_l2.append(idx_fname)
            dims.append(mod_dim[mod_name])
            offts_l3.append(str(offt_emb))
            offts_l3.append(str(offt_idx))
            offts_l3.append(str(len(timings)))
            if len(timings) > 0:
                duration = max(duration, timings.max())
        #fnames_l2 = [x.replace('/ssd/ssd_srv79/dumps/', 's3://bucket-7769-huanan/dza/dumps/') for x in fnames_l2]
            
        all_text = []
        all_start = []
        all_end = []
        
        cap_fname = os.path.join(out_root, str(vid)+'.capts')
        with open(cap_fname, 'w') as fout:
            fout.write('\t'.join(mod_names) + '\n')
            fout.write('\t'.join(map(str, dims)) + '\n')
            fout.write('\t'.join(fnames_l2) + '\n')
            fout.write('\t'.join(offts_l3) + '\n')
            used = set()
            for text, t_start, t_end in zip(mdata['text'], mdata['start'], mdata['end']):
                if t_end == -1:
                    t_end = duration
                if t_start >= t_end:
                    continue
                text = re.sub('\s+', ' ', text).strip()
                item = (text, t_start, t_end)
                if item in used:
                    continue
                #assert t_start < t_end
                offsets = []
                for mod_name in mod_names:
                    if mod_name in I:
                        embdim = mod_dim[mod_name]
                        emb_fname, idx_fname, offt_emb, offt_idx, timings, timings_offt = I[mod_name]
                        if t_start > timings[-1][1]:
                            # incorrect caption
                            break
                        if t_end < timings[0][0]:
                            # incorrect caption
                            break
                        #if vid == 'video2551':
                        #    import pdb; pdb.set_trace()
                        idx1 = find_segment_t0(timings, t_start)
                        idx2 = find_segment_t1(timings, t_end)+1
                        offsets.append(offt_emb + idx1 * embdim * 4)
                        offsets.append(timings_offt[idx1])
                        offsets.append(idx2 - idx1)
                    else:
                        offsets.append(-1)
                        offsets.append(-1)
                        offsets.append(0)
                    #import pdb; pdb.set_trace()
                if len(offsets)  < len(mod_names):
                    continue
                     
                joined_offsets = "\t".join(list(map(str, offsets)))
                fout.write(f'{text}\t{t_start}\t{t_end}\t{joined_offsets}\n')
                all_text.append(text)
                all_start.append(t_start)
                all_end.append(t_end)
        if len(all_text) == 0:
            os.remove(cap_fname)
        else:
            all_mdata[vid] = dict(text=all_text, start=all_start, end=all_end)
            processed.add(vid)



if __name__ == "__main__":
    main()
