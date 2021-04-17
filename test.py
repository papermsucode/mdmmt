import argparse
import os

import torch

from models.mmt import BertTXT, BertVID
from validate import validate
from data_loader.msrvtt_dataset import MSRVTT
from data_loader.activitynet_dataset import ActivityNet
from data_loader.lsmdc_dataset import LSMDC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--dataset_name', required=True, choices=[
        'MSRVTT_1kA',
        'MSRVTT_1kB',
        'MSRVTT_full',
        'lsmdc_publictest',
        'ActivityNet'])
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--video_batch_size', type=int, default=32)
    parser.add_argument('--text_batch_size', type=int, default=32)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    
    state = torch.load(args.checkpoint, map_location='cpu')

    print('Loading video model ...')
    experts_info = {
        'VIDEO': dict(dim=2048, idx=1, max_tok=30),
        'CLIP': dict(dim=512, idx=2, max_tok=30),
        'tf_vggish': dict(dim=128, idx=3, max_tok=30),
    }
    vid_bert_params = {
        'vocab_size_or_config_json_file': 10,
        'hidden_size': 512,
        'num_hidden_layers': 9,
        'intermediate_size': 3072,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.2,
        'attention_probs_dropout_prob': 0.2,
        'max_position_embeddings': 32,
        'type_vocab_size': 19,
        'initializer_range': 0.02,
        'layer_norm_eps': 1e-12,
        'num_attention_heads': 8,
    }
    model_vid = BertVID(expert_dims=experts_info, vid_bert_params=vid_bert_params)
    model_vid = model_vid.eval()
    model_vid.load_state_dict(state['vid_state_dict'])
    model_vid = model_vid.cuda()
    print('done')
    
    print('Loading text model ...')
    txt_bert_params = {
        'hidden_dropout_prob': 0.2,
        'attention_probs_dropout_prob': 0.2,
    }
    model_txt = BertTXT(
            modalities=list(experts_info.keys()),
            add_special_tokens=True,
            txt_bert_params=txt_bert_params,
    )
    model_txt = model_txt.eval()
    model_txt.load_state_dict(state['txt_state_dict'])
    model_txt = model_txt.cuda()
    print('done')
  

    print('Loading dataset ...') 
    dataset_name = args.dataset_name
    if dataset_name == 'MSRVTT_full':
        dataset = MSRVTT(
                    cut_name="full",
                    split_name="test",
                    data_dir=args.dataset_root,
                    experts_info=experts_info,
                    training=False)
    elif dataset_name == 'lsmdc_publictest':
        dataset = LSMDC(
                    cut_name="mrc",
                    split_name="test",
                    data_dir=args.dataset_root,
                    experts_info=experts_info,
                    training=False)
    elif dataset_name == 'ActivityNet':
        dataset = ActivityNet(
                    cut_name="mrc",
                    split_name="val",
                    data_dir=args.dataset_root,
                    experts_info=experts_info,
                    training=False)
    else:
        raise NotImplementedError(dataset_name)
        
    print('done')


    print('Validate ...')
    all_topk_t2v, all_topk_v2t = validate(
            video_model=model_vid,
            text_model=model_txt,
            dataset=dataset,
            embdim=3*512,
            rank=0,
            world_size=1,
            video_batch_size=args.video_batch_size,
            text_batch_size=args.text_batch_size,
            with_v2t=True)
    print('done')
          
    metrics = {}
    metrics["t2v/R1"] = 100 * float((all_topk_t2v==0).sum()) / len(all_topk_t2v)
    metrics["t2v/R5"] = 100 * float((all_topk_t2v < 5).sum()) / len(all_topk_t2v)
    metrics["t2v/R10"] = 100 * float((all_topk_t2v < 10).sum()) / len(all_topk_t2v)
    metrics["t2v/R50"] = 100 * float((all_topk_t2v < 50).sum()) / len(all_topk_t2v)
    metrics["t2v/MedR"] = all_topk_t2v.median().item() + 1
    metrics["t2v/MeanR"] = all_topk_t2v.mean().item() + 1
    metrics["v2t/R1"] = 100 * float((all_topk_v2t==0).sum()) / len(all_topk_v2t)
    metrics["v2t/R5"] = 100 * float((all_topk_v2t < 5).sum()) / len(all_topk_v2t)
    metrics["v2t/R10"] = 100 * float((all_topk_v2t < 10).sum()) / len(all_topk_v2t)
    metrics["v2t/R50"] = 100 * float((all_topk_v2t < 50).sum()) / len(all_topk_v2t)
    metrics["v2t/MedR"] = all_topk_v2t.median().item() + 1
    metrics["v2t/MeanR"] = all_topk_v2t.mean().item() + 1
    for key, val in metrics.items():
        print(f'{key}: {val}')



if __name__ == "__main__":
    main()



