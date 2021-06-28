# Introduction

In this repository we present the testing code for article 
[MDMMT: Multidomain Multimodal Transformer for Video Retrieval](https://arxiv.org/abs/2103.10699).

[Presentation](./MDMMT_HVU_CVPR_2021.pdf) from CVPR-2021 Workshop "[Large Scale Holistic Video Understanding](https://holistic-video-understanding.github.io/workshops/cvpr2021.html)".

This code helps:
1. Create embeddings with CLIP, irCSN152 and VGGish;
2. Create caption index files;
3. Run test with created embeddings and captions index files.

Our pretrained model is available [here](https://drive.google.com/file/d/1dVDouFZFEDrjqhvtjNFP633y0JjHf4Ya/view?usp=sharing).


# Citation
```
@misc{dzabraev2021mdmmt,
      title={MDMMT: Multidomain Multimodal Transformer for Video Retrieval}, 
      author={Maksim Dzabraev and Maksim Kalashnikov and Stepan Komkov and Aleksandr Petiushko},
      year={2021},
      eprint={2103.10699},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Expected testing results:

```
MSRVTT
t2v/R1: 22.87123745819398
t2v/R5: 49.67056856187291
t2v/R10: 61.66722408026756
t2v/R50: 83.95819397993311
t2v/MedR: 6.0
t2v/MeanR: 53.69550323486328
v2t/R1: 5.075250836120401
v2t/R5: 13.777591973244148
v2t/R10: 19.695652173913043
v2t/R50: 41.44314381270903
v2t/MedR: 84.0
v2t/MeanR: 695.3896484375


LSMDC
t2v/R1: 17.31731731731732
t2v/R5: 38.23823823823824
t2v/R10: 47.447447447447445
t2v/R50: 72.87287287287288
t2v/MedR: 12.0
t2v/MeanR: 59.398399353027344
v2t/R1: 16.716716716716718
v2t/R5: 37.73773773773774
v2t/R10: 45.545545545545544
v2t/R50: 72.27227227227228
v2t/MedR: 14.0
v2t/MeanR: 60.97697448730469

ActivityNet
t2v/R1: 19.673503557974048
t2v/R5: 45.22812892423608
t2v/R10: 56.7182921724571
t2v/R50: 80.8915864378401
t2v/MedR: 7.0
t2v/MeanR: 72.35956573486328
v2t/R1: 19.715362076182505
v2t/R5: 44.72582670573462
v2t/R10: 57.157806613645874
v2t/R50: 80.87065717873587
v2t/MedR: 7.0
v2t/MeanR: 68.22499084472656
``` 



# Downloads

```bash
mkdir -p ckpts
# https://github.com/facebookresearch/VMZ/
wget https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_ig65m_from_scratch_f125286141.pth -O ckpts/irCSN_152_ig65m_from_scratch_f125286141.pth

# https://github.com/tensorflow/models/tree/master/research/audioset/vggish
wget https://storage.googleapis.com/audioset/vggish_model.ckpt -O ckpts/vggish_model.ckpt

git clone https://github.com/openai/CLIP models/CLIP
git clone https://github.com/tensorflow/models/ models/tensorflow_models
```

# Environment
It is recommended to use conda to install packages.
It is recommended to create two environments. The first one for audio dumping, and the second for video

## Audio environment

Use this environment for producing embeddings with `tf_vggish`

```
tqdm
ffmpeg=4.2.2
tensorflow-gpu
tf_slim
resampy
six
pysoundfile
numpy=1.20.2 # !!! Make sure that intel-mkl is not used. It causes segfault in np.fft.rfft !!!
```

## Video environment

Use this environment for producing embeddings with `CLIP` and `irCSN152`

```bash
tqdm
pytorch=1.7.1 # !!! It is recommended to use pytorch=1.7.1; 1.8+ is not working with CLIP !!!
torchvision
ffmpeg=4.2.2
ftfy
regex
```



# Create lists

Replace `"<*_DATASET_ROOT>/` with directory where raw video files are located.

```bash
cat lists/LSMDC/fnames.lst | awk '{print "<LSMDC_DATASET_ROOT>/" $0}' > LSMDC.lst
cat lists/ActivityNet/fnames.lst | awk '{print "<ActivityNet_DATASET_ROOT>/" $0}' > ActivityNet_val.lst
cat lists/msrvtt/fnames.lst | awk '{print "<msrvtt_DATASET_ROOT>/" $0}' > msrvtt_test.lst
```


# Embeddings

```bash
python dumper.py \
    --model_type=VMZ_irCSN_152 \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/msrvtt/VMZ_irCSN_152/test  \
    --lst=msrvtt_test.lst \
    --nworker_per_gpu=2 \
    --per_batch_size=8 \
    --fps=32 \
    --frame_size=224 \
    --frame_crop_size=224 \
    --frames_per_clip=32

python dumper.py \
    --model_type=CLIP \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/msrvtt/CLIP/test  \
    --lst=msrvtt_test.lst \
    --nworker_per_gpu=8 \
    --per_batch_size=128 \
    --fps=1 \
    --frame_size=228 \
    --frame_crop_size=228 \
    --frames_per_clip=1

PYTHONPATH=\
models/tensorflow_models/research/audioset/vggish:\
models/tensorflow_models/research/audioset/:\
$PYTHONPATH \
python dumper.py \
    --model_type=tf_vggish \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/msrvtt/tf_vggish/test \
    --lst=msrvtt_test.lst \
    --nworker_per_gpu=2



python dumper.py \
    --model_type=VMZ_irCSN_152 \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/ActivityNet/VMZ_irCSN_152/test \
    --lst=ActivityNet_val.lst \
    --nworker_per_gpu=3 \
    --per_batch_size=8 \
    --fps=32 \
    --frame_size=224 \
    --frame_crop_size=224 \
    --frames_per_clip=32

python dumper.py \
    --model_type=CLIP \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/ActivityNet/CLIP/test \
    --lst=ActivityNet_val.lst \
    --nworker_per_gpu=3 \
    --per_batch_size=128 \
    --fps=1 \
    --frame_size=228 \
    --frame_crop_size=228 \
    --frames_per_clip=1 \
    --num_readers=8

PYTHONPATH=\
models/tensorflow_models/research/audioset/vggish:\
models/tensorflow_models/research/audioset/:\
$PYTHONPATH \
python dumper.py \
    --model_type=tf_vggish \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/ActivityNet/tf_vggish/test \
    --lst=ActivityNet_val.lst \
    --nworker_per_gpu=3 \
    --per_batch_size=32



python dumper.py \
    --model_type=VMZ_irCSN_152 \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/LSMDC/VMZ_irCSN_152/test \
    --lst=LSMDC.lst \
    --nworker_per_gpu=2 \
    --per_batch_size=8 \
    --fps=32 \
    --frame_size=224 \
    --frame_crop_size=224 \
    --frames_per_clip=32

python dumper.py \
    --model_type=CLIP \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/LSMDC/CLIP/test \
    --lst=LSMDC.lst \
    --nworker_per_gpu=2 \
    --per_batch_size=128 \
    --fps=1 \
    --frame_size=228 \
    --frame_crop_size=228 \
    --frames_per_clip=1 \
    --num_readers=8

PYTHONPATH=\
models/tensorflow_models/research/audioset/vggish:\
models/tensorflow_models/research/audioset/:\
$PYTHONPATH \
python dumper.py \
    --model_type=tf_vggish \
    --gpus=0,1,2,3,4,5,6,7 \
    --dst_prefix=/ssd/ssd_srv79/dza/dumps/LSMDC/tf_vggish/test \
    --lst=LSMDC.lst \
    --nworker_per_gpu=2 \
    --per_batch_size=32


```

# Create caption index

```bash
python create_capts.py \
	--dataset=msrvtt \
	--output_root=/tmp/capts/msrvtt/ \
	--modality=VIDEO:2048:/ssd/ssd_srv79/dza/dumps/msrvtt/VMZ_irCSN_152/test \
	--modality=CLIP:512:/ssd/ssd_srv79/dza/dumps/msrvtt/CLIP/test \
	--modality=tf_vggish:128:/ssd/ssd_srv79/dza/dumps/msrvtt/tf_vggish/test
mkdir -p /tmp/capts/msrvtt/symlinked-feats/
cp lists/msrvtt/test_list_full.txt /tmp/capts/msrvtt/symlinked-feats/test_list_full.txt

python create_capts.py \
	--dataset=ActivityNet \
	--output_root=/tmp/capts/ActivityNet/ \
	--modality=VIDEO:2048:/ssd/ssd_srv79/dza/dumps/ActivityNet/VMZ_irCSN_152/test \
	--modality=CLIP:512:/ssd/ssd_srv79/dza/dumps/ActivityNet/CLIP/test \
	--modality=tf_vggish:128:/ssd/ssd_srv79/dza/dumps/ActivityNet/tf_vggish/test
mkdir -p /tmp/capts/ActivityNet/symlinked-feats/
cp lists/ActivityNet/val.vids /tmp/capts/ActivityNet/symlinked-feats/val.vids

python create_capts.py \
	--dataset=lsmdc_publictest \
	--output_root=/tmp/capts/LSMDC/ \
	--modality=VIDEO:2048:/ssd/ssd_srv79/dza/dumps/LSMDC/VMZ_irCSN_152/test \
	--modality=CLIP:512:/ssd/ssd_srv79/dza/dumps/LSMDC/CLIP/test \
	--modality=tf_vggish:128:/ssd/ssd_srv79/dza/dumps/LSMDC/tf_vggish/test
mkdir -p /tmp/capts/LSMDC/symlinked-feats/
cp lists/LSMDC/test.vids /tmp/capts/LSMDC/symlinked-feats/test.vids
```


# Test

```bash
python test.py --dataset_root=/tmp/capts/msrvtt/  --checkpoint=<PATH_TO_MODEL>/mdmmt_3mod.pth  --dataset_name=MSRVTT_full --gpu=2
python test.py --dataset_root=/tmp/capts/LSMDC/  --checkpoint=<PATH_TO_MODEL>/mdmmt_3mod.pth  --dataset_name=lsmdc_publictest --gpu=2
python test.py --dataset_root=/tmp/capts/ActivityNet/  --checkpoint=<PATH_TO_MODEL>/mdmmt_3mod.pth  --dataset_name=ActivityNet --gpu=2

```

# WARNING

Do not use numpy with mkl backend. Sometimes np.fft.rfft produce segmentation fault.
