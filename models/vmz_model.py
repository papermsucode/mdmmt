import torch
import numpy as np


from models.pt_utils import IdLayer
from models.vmz.r2plus1d import r2plus1d_152, r2plus1d_34
from models.vmz.csn import ip_csn_152, ir_csn_152


class VMZ_base:
    def __init__(self, ckpt_path, model_cls):
        with torch.no_grad():
            self.model = model = model_cls()
            model.fc = IdLayer()
            model.eval()
            state_dict = torch.load(ckpt_path)
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            model.load_state_dict(state_dict)
            model.cuda()
            self.mean = torch.tensor([0.43216, 0.394666, 0.37645])[None,:,None,None,None].cuda()
            self.std = torch.tensor([0.22803, 0.22145, 0.216989])[None,:,None,None,None].cuda()
    @torch.no_grad()
    def __call__(self, batch_imgs):
        # (bs, t, h, w, c) each pixel in [0,255]
        imgs = torch.from_numpy(batch_imgs.astype(np.float32)).cuda()
        imgs.div_(255.)
        imgs = imgs.permute(0, 4, 1, 2, 3)
        imgs = (imgs - self.mean) / self.std
        embs = self.model(imgs)
        embs = embs.cpu().numpy()
        return embs

class VMZ_r2plus1d_152(VMZ_base):
    def __init__(self, ckpt_path):
        super().__init__(ckpt_path, r2plus1d_152)

class VMZ_r2plus1d_34(VMZ_base):
    def __init__(self, ckpt_path):
        super().__init__(ckpt_path, r2plus1d_34)

class VMZ_ipCSN_152(VMZ_base):
    def __init__(self, ckpt_path):
        super().__init__(ckpt_path, ip_csn_152)

class VMZ_irCSN_152(VMZ_base):
    def __init__(self, ckpt_path):
        super().__init__(ckpt_path, ir_csn_152)


