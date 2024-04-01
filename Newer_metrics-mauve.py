import glob
import os
import json
import pandas as pd 
from Generation.eval import *
from transformers import AutoTokenizer,AutoModelForCausalLM
import argparse
import mauve 
import random

path_datasets = 'Datasets/'
path_result   = '../../../HULK_new/Counterspeech/Results/'
save_path     = 'Results/'


def mauve_score(params):
    #https://github.com/krishnap25/mauve
    hypo = params[0]  # a list of generated_hypothesis   
    refs = params[1]  # a list of refrences for particular_refrences    
    
    p_text = hypo
    q_text = refs
    print(q_text[0:5])
    print(len(p_text),len(q_text))
    
    out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0,mauve_scaling_factor=1,featurize_model_name='gpt2-large',max_text_length=100, verbose=False)
    return out.mauve



def mappings(gen_dict,ref):
    mp = {}
    for key in ref.keys():
        ref_ht = ref[key]["hatespeech"]
        for key2 in gen_dict['samples'].keys():
            gen_ht = gen_dict['samples'][key2]["org_hate"]
            if gen_ht==ref_ht:
                mp[key2]   = key
    return mp

def heavy_metrics(train_sample,gen_sample,metric):
    train_path = glob.glob(path_datasets+'/'+train_sample+'/Train.csv')[0]
    print(train_path)
    gen        = glob.glob(save_path+'/**'+gen_sample+'_huggingface_base.json')
    gen = sorted(gen,key = os.path.getmtime)
    print(path_datasets+'/'+train_sample+'/Test.json')
    ref        = glob.glob(path_datasets+'/'+train_sample+'/Test.json')[0]
    ref_csv  = pd.read_json(ref)
    ref_dict={}
    for index,row in ref_csv.iterrows():
        ref_dict[index]={'hatespeech':row['initiator_message'],'counterspeech':row['reply_message']}
    
    
    scores={}
    for item in gen:
        print(item)
    it = 1
    for files in gen:
        with open(files, 'r') as file:
            gen_dict  = json.loads(file.read())
        hypo = []
        refs = []
        mp = mappings(gen_dict,ref_dict)
        for key in gen_dict['samples']:
            if key in mp.keys():  
                sentences=gen_dict['samples'][key]['counterspeech_model']
                cs_refs = ref_dict[mp[key]]['counterspeech']
                
                min_length = min(len(sentences),len(cs_refs))
                hypo+=random.sample(sentences,min_length)
                refs+=random.sample(cs_refs,min_length)
    
        train = pd.read_csv(train_path)
        train_set = list(zip(train['initiator_message'].tolist(),train['reply_message'].tolist()))
        params = [hypo,refs]
        
        final_score=mauve_score(params)
        data_dict = {
                     metric:final_score
                     }
        key = files
        print(key.split('/Results/'))
        scores[key.split('/Results/')[0]] = data_dict
        print(f'Key:{key}')
        for d in data_dict:
            print(d+ ':' + str(data_dict[d]))
        print("===============================")
        it +=1 
    write_in='Metrics_eval_results/Newer_metrics_'+metric+'_on_'+train_sample+".json"
    with open(write_in, 'w') as outfile:
         json.dump(scores, outfile,indent=4)








if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('dataset',
                           metavar='--d',
                           type=str,
                           help='location where the dataset is saved')
    
#     my_parser.add_argument('metric',
#                            metavar='--m',
#                            type=str,
#                            help='which metric to be used')
    
    
    
    device='cuda'
    if torch.cuda.is_available() and device=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
            deviceID = [1]
            #deviceID =get_gpu(gpu_id)
#            torch.cuda.set_device(1)
            ##### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
            torch.cuda.set_device(deviceID[0])

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    args = my_parser.parse_args()
    dataset=args.dataset
    metric='mauve'
    
    heavy_metrics(dataset,dataset,metric)
    
    
    