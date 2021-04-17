import os
import torch
import numpy as np
import re

def np_choice(rnd, arr):
    if rnd:
        choice_fn = rnd.choice
    else:
        choice_fn = np.random.choice
    idx = choice_fn(len(arr))
    return arr[idx]


class RangeBasedMixin:
    def __init__(self, **kwargs):
        self.fname_repl = kwargs.pop('fname_repl', None)
        self.capt_paths = {}
        self.capts_root = os.path.join(self.data_dir, 'capts')
        assert os.path.exists(self.capts_root), self.capts_root
        for root, dnames, fnames in os.walk(self.capts_root):
            for fname in fnames:
                if fname.endswith('.capts'):
                    vid = os.path.splitext(fname)[0]
                    path = os.path.join(self.capts_root, fname)
                    self.capt_paths[vid] = path
        self.fidx_size = {} # fname --> size of file
        self.experts = list(self.experts_info.keys())
        super().__init__(**kwargs)

    def read_timings(self, fname, offt, nemb, blksz=4096):
        if fname not in self.fidx_size:
            fsize = os.path.getsize(fname)
            self.fidx_size[fname] = fsize
        else:
            fsize = self.fidx_size[fname]
        fname_or_f = open(fname)

        timings = []
        buf = ''
        while len(timings) < nemb and offt < fsize:
            rsz = min(blksz, fsize - offt)
            b = self.read_range(fname_or_f, offt, rsz)
            if type(b) is bytes:
                b = b.decode('utf8')
            buf = buf + b
            offt += rsz
            idx = 0
            while True:
                idx1 = buf.find('\n', idx)
                if idx1 == -1:
                    if idx > 0:
                        buf = buf[idx:]
                    break
                if idx1 == fsize - 1:
                    break
                a, b = buf[idx: idx1].split('\t')
                idx = idx1 + 1
                timings.append((float(a), float(b)))
                if len(timings) == nemb:
                    break
        if type(fname_or_f) is not str:
            fname_or_f.close()
        return np.array(timings)

    def read_embds(self, fname, offt, nemb, embdim):
        bsz = nemb * embdim * 4
        data = self.read_range(fname, offt, bsz)
        embds = np.frombuffer(data, dtype=np.float32).reshape(-1, embdim)
        return embds

    def read_range(self, fname_or_f, offt, size):
        if type(fname_or_f) is str:
            with open(fname_or_f, 'rb') as f:
                f.seek(offt)
                data = f.read(size)
        else:
            fname_or_f.seek(offt)
            data = fname_or_f.read(size)
        return data

    def parse_line(self, line):
        s = line.strip().split('\t')
        text = s[0]
        t_start = float(s[1])
        t_end = float(s[2])
        s1 = list(map(int, s[3:]))
        embd_offts = s1[0::3]
        idx_offts = s1[1::3]
        nembs = s1[2::3]
        return text, t_start, t_end, embd_offts, idx_offts, nembs

    def get_sample_data(self, vid, capidx=None, caponly=False, seed=None):
        # -1 return random caption and corresponding embeddings
        # >=0 return one caption with index=capt_Idx and corresponding embeddings
        # None return all embds and capts
        if seed is not None:
            rnd = np.random.RandomState(seed)
        else:
            rnd = None
        choice_fn = lambda arr: np_choice(rnd, arr)

        path = self.capt_paths[vid]
        captions = []
        captions_t = []
        captions_idxs = []
        features = {}
        features_t = {}
        with open(path) as f:
            mod_names = f.readline().strip().split('\t')
            _mod_dims = f.readline().strip().split('\t')
            mod_dim = {mod_name: int(dim) for mod_name, dim in zip(mod_names, _mod_dims)}
            _embd_idx_fnames = f.readline()[:-1].split('\t')
            if self.fname_repl:
                a, b = self.fname_repl
                _embd_idx_fnames = [re.sub(a, b, x) for x in _embd_idx_fnames]
            embd_fnames = _embd_idx_fnames[0::2]
            idx_fnames = _embd_idx_fnames[1::2]
            _offt_nemb = f.readline().strip().split('\t')
            global_embd_offts = list(map(int, _offt_nemb[0::3]))
            global_idx_offts = list(map(int, _offt_nemb[1::3]))
            global_nembs = list(map(int, _offt_nemb[2::3]))

            assert len(mod_names) == len(embd_fnames)
            assert len(mod_names) == len(idx_fnames)
            assert len(mod_names) == len(global_embd_offts)
            assert len(mod_names) == len(global_idx_offts)
            assert len(mod_names) == len(global_nembs)

            lines = list(enumerate(f))
            if capidx is not None:
                if capidx >= 0:
                    selected_cap_idx, capt_line = lines[capidx]
                    capt_line = capt_line.strip()
                else:
                    selected_cap_idx, capt_line = choice_fn(lines)
                    capt_line = capt_line.strip()
                text, t0, t1, embd_offts, idx_offts, nembs = self.parse_line(capt_line)
                captions_t.append((t0, t1))
                captions.append(text)
                captions_idxs.append(selected_cap_idx)
                if not caponly:
                    for mod_name, embd_fname, idx_fname, offt_embd, offt_idx, nemb in zip(mod_names, embd_fnames, idx_fnames, embd_offts, idx_offts, nembs):
                        if mod_name in self.experts and nemb > 0:
                            embdim = mod_dim[mod_name]
                            max_tok = self.experts_info[mod_name]["max_tok"]
                            nemb = min(max_tok, nemb)
                            features[mod_name] = self.read_embds(embd_fname, offt_embd, nemb, embdim)
                            features_t[mod_name] = self.read_timings(idx_fname, offt_idx, nemb)
                            assert len(features[mod_name]) == len(features_t[mod_name]), (mod_name, embd_fname, idx_fname, offt_embd, offt_idx, nemb, vid, capidx, len(features[mod_name]), len(features_t[mod_name]))
            else:
                # read all
                for cap_idx, line in lines:
                    text, t0, t1, _embd_offts, _idx_offts, _nembs = self.parse_line(line)
                    captions.append(text)
                    captions_t.append((t0, t1))
                    captions_idxs.append(cap_idx)
                if not caponly:
                    for mod_name, embd_fname, idx_fname, offt_embd, offt_idx, nemb in zip(mod_names, embd_fnames, idx_fnames, global_embd_offts, global_idx_offts, global_nembs):
                        if mod_name in self.experts and nemb > 0:
                            embdim = mod_dim[mod_name]
                            max_tok = self.experts_info[mod_name]["max_tok"]
                            nemb = min(max_tok, nemb)
                            features[mod_name] = self.read_embds(embd_fname, offt_embd, nemb, embdim)
                            features_t[mod_name] = self.read_timings(idx_fname, offt_idx, nemb)
                            assert len(features[mod_name]) == len(features_t[mod_name]), (vid, capidx)

        if caponly:
            return dict(captions=captions,
                        captions_t=captions_t,
                        captions_idxs=captions_idxs)
        else:
            return dict(captions=captions,
                        captions_t=captions_t,
                        captions_idxs=captions_idxs,
                        features=features,
                        features_t=features_t)

