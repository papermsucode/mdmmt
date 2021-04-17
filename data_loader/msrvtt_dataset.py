# Copyright 2020 Valentin Gabeur
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MSR-VTT dataset."""
import os
import os.path

from data_loader.base import BaseDataset
import numpy as np

def get_val_captions(self, cut_name,  **kwargs):
    all_captions = []
    for vid in self.vid_list:
        if cut_name == 'jsfusion':
            capidx = self.restrict_test_captions[vid]
        elif cut_name == 'miech':
            capidx = 0
        else:
            raise NotImplementedError
        sample_data = self.get_sample_data(vid, capidx=capidx, caponly=True)
        captions = sample_data["captions"]
        captions_t = sample_data["captions_t"]
        captions_idxs = sample_data["captions_idxs"]
        assert captions_idxs[0] == capidx
        assert len(captions) == len(captions_t) == len(captions_idxs), (len(captions), len(captions_t), len(captions_idxs))

        for capidx, cap, (t0, t1) in zip(captions_idxs, captions, captions_t):
            all_captions.append((vid, capidx, cap, t0, t1))
    return all_captions

class MSRVTT(BaseDataset):
  """MSR-VTT dataset."""

  def configure_train_test_splits(self, cut_name, split_name):
    self.restrict_test_captions = None

    if cut_name in ["miech", "jsfusion"]:
      self.get_val_captions = lambda *args, **kwargs: get_val_captions(self, cut_name, *args, **kwargs)
      if cut_name in ["miech"]:
        # For now, we follow Antoine's approach of using the first text caption
        # for the retrieval task when evaluating on his custom split.
        train_list_path = "train_list_miech.txt"
        test_list_path = "test_list_miech.txt"
      elif cut_name in ["jsfusion"]:
        train_list_path = "train_list_jsfusion.txt"
        test_list_path = "val_list_jsfusion.txt"
        # NOTE: The JSFusion split (referred to as 1k-A in the paper) uses all
        # videos, but randomly samples a single caption per video from the test
        # set for evaluation. To reproduce this evaluation, we use the indices
        # of the test captions, and restrict to this subset during eval.
        test_cap_idx_path = os.path.join(self.data_dir, "symlinked-feats",
                                         "jsfusion_val_caption_idx.pkl")
        with open(test_cap_idx_path, 'rb') as f:
          self.restrict_test_captions = pickle.load(f)

      test_list_path = os.path.join(self.data_dir, "symlinked-feats", test_list_path)
      with open(test_list_path) as f:
        test_vid_list = f.readlines()
      nb_test_samples = len(test_vid_list)

      if split_name in ["train", "trn", "val", "trainval"]:
        train_list_path = os.path.join(self.data_dir, "symlinked-feats", train_list_path)
        with open(train_list_path) as f:
          train_vid_list = f.readlines()
        nb_train_samples = len(train_vid_list)

        cross_vid_list = train_vid_list
        cross_vid_list = [x.strip() for x in cross_vid_list]

        # The cross seed is used to split training videos into different
        # cross validation splits.
        rng = np.random.RandomState(0)
        rng.shuffle(cross_vid_list)

        if split_name in ["train", "trn", "trainval"]:
          if split_name in ["trainval"]:
            self.vid_list = cross_vid_list
          elif split_name in ["train", "trn"]:
            self.vid_list = cross_vid_list[nb_test_samples:]
          if split_name in ["trn"]:
            self.vid_list = self.vid_list[:nb_test_samples]

        elif split_name in ["val"]:
          self.vid_list = cross_vid_list[:nb_test_samples]

      elif split_name == "test":
        self.vid_list = test_vid_list
        self.vid_list = [x.strip() for x in self.vid_list]

    elif cut_name in ["full", 'full_clean']:
      if split_name in ["train", "trn"]:
        #list_path = "train_list.txt"
        if cut_name == 'full':
          list_path = "symlinked-feats/train_list_full.txt"
        else:
          list_path = "symlinked-feats/train_list_full.ytvid.manual.txt"
      elif split_name in ["val"]:
        #list_path = "val_list.txt"
        list_path = "symlinked-feats/test_list_full.txt"
      elif split_name in ["test"]:
        list_path = "symlinked-feats/test_list_full.txt"
      else:
        raise ValueError(f"unrecognised split: {split_name}")
      list_path = os.path.join(self.data_dir, list_path)
      with open(list_path) as f:
        self.vid_list = f.readlines()
      self.vid_list = [x.strip() for x in self.vid_list]

      # We want the trn split to be the same size as the val set
      if split_name in ["trn"]:
        rng = np.random.RandomState(0)
        rng.shuffle(self.vid_list)
        self.vid_list = self.vid_list[:497]
    else:
      msg = "unrecognised cut: {}"
      raise ValueError(msg.format(cut_name))

    self.split_name = split_name
    self.dataset_name = f"MSRVTT_{cut_name}_{split_name}"
    
