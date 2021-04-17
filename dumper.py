import os
import argparse
import subprocess
import threading
import queue

from tqdm import tqdm
import numpy as np

from mp_utils import MpGen


import multiprocessing as mp
import signal
import traceback
import sys



def proc_pack(input_it, dst_prefix):
    print('Started proc_pack')
    emb_fname = dst_prefix+'.emb'
    idx_fname = dst_prefix+'.idx'
    dname = os.path.dirname(emb_fname)
    os.makedirs(dname, exist_ok=True)
    with open(emb_fname, 'wb') as fout_emb, open(idx_fname, 'w') as fout_idx:
        for video_path, timings, embs in input_it:
            if embs is None:
                yield video_path
                continue
            assert len(embs) == len(timings), (len(embs), len(timings))
            fout_emb.write(embs.tobytes())
            fout_idx.write(f'VIDEO\t{video_path}\n')
            for a, b in timings:
                fout_idx.write(f'{a:.2f}\t{b:.2f}\n')
            yield video_path

def read_frames(video_path, fps, frame_size, frame_crop_size, alpha_h, alpha_w, hflip=False, max_frames=None):
    # 1. scale to frame_size on short side
    # 2. crop frame_crop_size
    #
    # alpha_h, alpha_w is used for control crop position after rescale
    # alpha_h=0.5, alpha_w=0.5 is equivalent to center crop
    # alpha_h=1, alpha_w=1 is equivalent to most right most bottom crop
    assert 0 <= alpha_h <= 1, alpha_h
    assert 0 <= alpha_w <= 1, alpha_w
 
    scale_w = f'round((iw/min(iw\,ih)*{frame_size})/2)*2'
    scale_h = f'round((ih/min(iw\,ih)*{frame_size})/2)*2'
    w0 = f'round(({scale_w}-{frame_crop_size})*{alpha_w})'
    h0 = f'round(({scale_h}-{frame_crop_size})*{alpha_h})'
    crop_w = f'{frame_crop_size}'
    crop_h = f'{frame_crop_size}'
    hflip_filter = ',hflip' if hflip else ''
    dframes = f'-dframes {max_frames}' if max_frames else ''
 
    cmd = f"ffmpeg -y -i {video_path}  -max_muxing_queue_size 9999  -loglevel error -vf 'fps={fps}:round=up,scale={scale_w}:{scale_h},crop={crop_w}:{crop_h}:{w0}:{h0}{hflip_filter}' {dframes} -pix_fmt rgb24 -f rawvideo -nostdin pipe:"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    while True:
        data = p.stdout.read(10000 * frame_crop_size*frame_crop_size*3)
        if not data:
            break
        yield data


def read_frames_center_crop(video_path, fps, frame_size, frame_crop_size):
    return read_frames(video_path, fps, frame_size, frame_crop_size, alpha_h=0.5, alpha_w=0.5)

def read_frames_center_crop_batch(video_path, fps, frame_size, frame_crop_size, batch_num_frames):
    batch_byte_size = batch_num_frames * frame_crop_size * frame_crop_size * 3
    data0 = b''
    for data in read_frames_center_crop(video_path, fps, frame_size, frame_crop_size):
        if len(data0) > 0:
            data0 = data0 + data
        else:
            data0 = data
        while len(data0) > batch_byte_size:
            data_batch = data0[:batch_byte_size]
            data0 = data0[batch_byte_size:]
            frames = np.frombuffer(data_batch, dtype=np.uint8).reshape(-1, frame_crop_size, frame_crop_size, 3) # (nframes, h, w, c)
            yield frames
    if len(data0):
        frames = np.frombuffer(data0, dtype=np.uint8).reshape(-1, frame_crop_size, frame_crop_size, 3)
        yield frames

def ffmpeg_audio_reader(in_filename):
    cmd = f'ffmpeg -i {in_filename} -loglevel quiet  -f wav -nostdin  pipe:'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    all_data = b''
    while True:
        data = p.stdout.read(1024**2)
        if not data:
            break
        all_data += data
    if len(all_data) == 0:
        return None
    else:
        return all_data

class AudioDecoder:
    def __init__(self, input_it, num_workers=2):
        self.input_it = input_it
        self.workers = []
        self.q = queue.Queue()
        self.num_running_workers = num_workers
        import warnings
        warnings.filterwarnings("ignore")
        for _ in range(num_workers):
            th = threading.Thread(target=self.worker_decoder)
            th.start()
            self.workers.append(th)
    
    def worker_decoder(self):
        import vggish.vggish_input as vggish_input
        import io
        import scipy.io.wavfile as scio

        for path in self.input_it:
            wav = ffmpeg_audio_reader(path)
            if wav is None:
                # no audio channel
                timings, segms = None, None
            else:
                sr, data = scio.read(io.BytesIO(wav))
                data = data / 32768.0
                segms = vggish_input.waveform_to_examples(data, sr)
                t_start = np.arange(len(segms), dtype=np.float32) * 0.96
                t_end = t_start + 0.96
                timings = np.concatenate([t_start[..., None], t_end[..., None]], axis=1) # (nsegm, 2)
            self.q.put((path, timings, segms))
        self.q.put(None)
    
    def __iter__(self):
        return self

    def __next__(self):
        while True:
            data = self.q.get()
            if data is None:
                self.num_running_workers -= 1
                if self.num_running_workers == 0:
                    for th in self.workers:
                        th.join()
                    raise StopIteration
                continue
            return data


def proc_dumper_video_1(
        input_it,
        fps,
        frame_size,
        frame_crop_size,
        frames_per_clip,
        per_batch_size,
        model,
        lock,
        q_out):
    for path in input_it:
        frames_batch_iter = read_frames_center_crop_batch(
                video_path=path,
                fps=fps,
                frame_size=frame_size,
                frame_crop_size=frame_crop_size,
                batch_num_frames=per_batch_size*frames_per_clip)
        # frames_batch_iter: (-1, h, w, c)
        embs = []
        timings = []
        t = 0
        delta = frames_per_clip / fps
        for frames in frames_batch_iter:
            if len(frames) % frames_per_clip > 0:
                n = len(frames)
                n1 = int(len(frames) // frames_per_clip * frames_per_clip)
                frames1 = frames[:n1]
                # increase frame rate in the last video segment
                idxs = np.ceil(np.linspace(n1, n-1, frames_per_clip)).astype(np.long)
                frames2 = frames[idxs]
                frames = np.concatenate([frames1, frames2], axis=0)
            assert len(frames) % frames_per_clip == 0
            batch_frames = frames.reshape(-1, frames_per_clip, frame_crop_size, frame_crop_size, 3)
            for _ in range(len(batch_frames)):
                timings.append((t, t + delta))
                t += delta
            with lock:
                embs.append(model(batch_frames))
        if len(embs) > 0:
            embs = np.concatenate(embs, axis=0)
            timings = np.array(timings) # (nsegm, 2)
        else:
            print(f'Nothing decoded: {path}')
            embs = None
            timings = None
        q_out.put((path, timings, embs))
    q_out.put(None)

    
def proc_dumper_video(
        input_it,
        gpu,
        fps,
        frame_size,
        frame_crop_size,
        frames_per_clip,
        model_type,
        per_batch_size,
        num_readers=2):
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    if model_type == 'VMZ_irCSN_152':
        from models.vmz_model import VMZ_irCSN_152
        model = VMZ_irCSN_152('ckpts/irCSN_152_ig65m_from_scratch_f125286141.pth')
    elif model_type == 'CLIP':
        from models.clip_model import CLIP
        model = CLIP()
    else:
        raise NotImplementedError
    lock = threading.Lock()
    q = queue.Queue(20)
    threads = []
    for _ in range(num_readers):
        th = threading.Thread(target=proc_dumper_video_1, kwargs=dict(
            input_it=input_it,
            fps=fps,
            frame_size=frame_size,
            frame_crop_size=frame_crop_size,
            frames_per_clip=frames_per_clip,
            per_batch_size=per_batch_size,
            model=model,
            lock=lock,
            q_out=q))
        th.start()
        threads.append(th)
    num_alive = num_readers
    while num_alive > 0:
        x = q.get()
        if x is None:
            num_alive -= 1
            continue
        yield x
    for th in threads:
        th.join()



def proc_dumper_audio(
        input_it,
        gpu,
        model_type,
        per_batch_size):
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    if model_type == 'tf_vggish':
        #print(f'DEVICE: {gpu}')
        #import tensorflow as tf
        #device = tf.config.list_physical_devices('GPU')[gpu]
        #tf.config.set_visible_devices([device], 'GPU')
        from models.vggish_model import VGGish
        model = VGGish('ckpts/vggish_model.ckpt', per_batch_size=per_batch_size)
    else:
        raise NotImplementedError

    loader = AudioDecoder(input_it=input_it, num_workers=1)
    for path, timings, frames in loader:
        if frames is None:
            yield path, timings, None
            continue
        embs = []
        idxs = range(0, len(frames), per_batch_size)
        for idx in idxs:
            batch = frames[idx: idx + per_batch_size]
            embs.append(model(batch))
        if len(embs) > 0:
            embs = np.concatenate(embs, axis=0)
            yield path, timings, embs
        else:
            yield path, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['CLIP', 'VMZ_irCSN_152', 'tf_vggish'])
    parser.add_argument('--gpus', type=lambda x: list(map(int, x.split(','))), default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--dst_prefix', required=True)
    parser.add_argument('--lst', help='each line is path to video file', required=True)
    parser.add_argument('--nworker_per_gpu', type=int, default=4)
    parser.add_argument('--num_readers', type=int, default=2)
    parser.add_argument('--per_batch_size', type=int, default=8)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--frame_size', type=int, default=256)
    parser.add_argument('--frame_crop_size', type=int, default=224)
    parser.add_argument('--frames_per_clip', type=int, default=30)
    args = parser.parse_args()

    lst = [] # paths to video files
    with open(args.lst) as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            lst.append(path)

    g0 = lst
    
    num_workers_dumper = len(args.gpus)*args.nworker_per_gpu
    if args.model_type in ['CLIP', 'VMZ_irCSN_152']:
        proc_dumper_fn = lambda input_it, rank: proc_dumper_video(
                input_it=input_it,
                gpu=args.gpus[rank % len(args.gpus)],
                fps=args.fps,
                frame_size=args.frame_size,
                frame_crop_size=args.frame_crop_size,
                frames_per_clip=args.frames_per_clip,
                model_type=args.model_type,
                per_batch_size=args.per_batch_size,
                num_readers=args.num_readers)
    elif args.model_type in ['tf_vggish']:
        proc_dumper_fn = lambda input_it, rank: proc_dumper_audio(
                input_it=input_it,
                gpu=args.gpus[rank % len(args.gpus)],
                model_type=args.model_type,
                per_batch_size=args.per_batch_size)
    g1 = MpGen(g0,
            proc_dumper_fn,
            num_workers=num_workers_dumper,
            streaming_mode=True)

    proc_pack_fn = lambda input_it, rank: proc_pack(
            input_it=input_it,
            dst_prefix=args.dst_prefix)    
    g2 = MpGen(g1, proc_pack_fn, num_workers=1, streaming_mode=True)

    for _ in tqdm(g2, total=len(g0)):
        pass



if __name__ == "__main__":
    main()
