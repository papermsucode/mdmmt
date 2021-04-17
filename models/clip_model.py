import torch
import numpy as np
import models.CLIP.clip as clip

class CLIP:
    def __init__(self):
        self.device = "cuda"
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None,:,None,None].cuda()
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None,:,None,None].cuda()
 
    @torch.no_grad()
    def __call__(self, frames):
        # frames: (bs, t, h, w, c), t=1
        frames = torch.from_numpy(frames.astype(np.float32)).squeeze(dim=1).to(self.device) # (bs, h, w, c)
        frames.div_(255.)
        frames = frames.permute(0, 3, 1, 2)
        frames = (frames - self.mean) / self.std
        embs = self.model.encode_image(frames)
        return embs.cpu().numpy().astype('float32')


