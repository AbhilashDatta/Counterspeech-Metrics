import glob
import os
import json
import pandas as pd 
from Generation.eval_new import *
from transformers import AutoTokenizer,AutoModelForCausalLM
import argparse
import pickle as pkl

path_datasets = 'Datasets/'
path_result   = '../../../HULK_new/Counterspeech/Results/'
save_path     = 'Results/'

def mappings(gen_dict,ref):
    mp = {}
    for key in ref.keys():
        ref_ht = ref[key]["hatespeech"]
        for key2 in gen_dict['samples'].keys():
            gen_ht = gen_dict['samples'][key2]["org_hate"]
            if gen_ht==ref_ht:
                mp[key2]   = key
    return mp

def mappings_baseline(gen_dict,ref):
    mp = {}
    for key in ref.keys():
        ref_ht = ref[key]["hatespeech"]
        for key2 in gen_dict.keys():
            gen_ht = gen_dict[key2]["hatespeech"]
            if gen_ht==ref_ht:
                mp[key2]   = key
    return mp

def heavy_metrics(train_sample,gen_sample,metric):
#     gen        = glob.glob(save_path+'/**'+gen_sample+'_huggingface_base.json')
#     gen2        = glob.glob(save_path+'/**'+gen_sample+'_topic_argument_huggingface_base.json')
#     gen=gen+gen2
    gen5         =  glob.glob(save_path+'/'+'*'+gen_sample+'*_huggingface_base.json')
    gen = gen5
    gen = sorted(gen,key = os.path.getmtime)
    print(path_datasets+'/'+train_sample+'/Test.json')
    ref        = glob.glob(path_datasets+'/'+train_sample+'/Test.json')[0]
    ref_csv  = pd.read_json(ref)
    ref_dict={}
    for index,row in ref_csv.iterrows():
        ref_dict[index]={'hatespeech':row['initiator_message'],'counterspeech':row['reply_message']}
    

#     with open(ref, 'r') as file:
#         ref_dict   = json.loads(file.read())    
#     print(ref_dict)
    for item in gen:
        print(item)
    it = 1
    if(metric=='bleurt'):
        bleurt_score=Bleurt(model_path="Elron/bleurt-base-512",cache_path='../../Saved_models',max_length=200, batch_size=16,use_gpu=True)
    elif(metric=='argument'):
        argument_score=Argument_scoring(model_path='chkla/roberta-argument',cache_path='../../Saved_models',max_length=100, batch_size=16,use_gpu=True)
    elif(metric=='counterspeech'):
        counterspeech_score=Argument_scoring(model_path='Saved_Models/Discriminator/Counterspeech_bert-base-uncased',cache_path='../../Saved_models',max_length=100, batch_size=16,use_gpu=True)
    elif(metric=='upvote_prob'):
        dialog_upvote=Dialog_upvote_scoring(model_path='microsoft/DialogRPT-updown',cache_path='../../Saved_models',max_length=100, batch_size=16,use_gpu=True)
    elif(metric=='width_prob'):
        dialog_width=Dialog_upvote_scoring(model_path='microsoft/DialogRPT-width',cache_path='../../Saved_models',max_length=100, batch_size=16,use_gpu=True)
    elif(metric=='depth_prob'):
        dialog_depth=Dialog_upvote_scoring(model_path='microsoft/DialogRPT-depth',cache_path='../../Saved_models',max_length=100, batch_size=16,use_gpu=True)
    elif(metric=='counter_argument'):
        counter_argument_score=Counter_argument_scoring(model_path='Saved_Models/Discriminator/Argument_bert-base-uncased',cache_path='../../Saved_models',max_length=200, batch_size=8,use_gpu=True)
    elif(metric=='toxicity'):
        toxicity_score=Toxic_HateXplain_scoring(model_path=None,cache_path='../../Saved_models',max_length=100, batch_size=16,use_gpu=True)
    
    
    scores={}
    
    
    
    for files in gen:
        with open(files, 'r') as file:
            gen_dict  = json.loads(file.read())
        hypo = []
        refs = []
        hate_inst=[]
        mp = mappings(gen_dict,ref_dict)
        for key in gen_dict['samples']:
            if key in mp.keys():  
                for sentences in gen_dict['samples'][key]['counterspeech_model']:
                    hypo.append(sentences)
                    cs_refs = ref_dict[mp[key]]['counterspeech']
                    refs.append(cs_refs)
                    hate_inst.append(ref_dict[mp[key]]['hatespeech'])
    
        params = [hypo,refs]
        
        if(metric=='bleurt'):
            final_score=bleurt_score.score(params)[1]
        elif(metric=='argument'):
            final_score=argument_score.scoring(hypo)[1]
        elif(metric=='counterspeech'):
            final_score=counterspeech_score.scoring(hypo)[1]
        elif(metric=='upvote_prob'):
            final_score=dialog_upvote.scoring(hypo,hate_inst)[1]
        elif(metric=='width_prob'):
            final_score=dialog_width.scoring(hypo,hate_inst)[1]
        elif(metric=='depth_prob'):
            final_score=dialog_depth.scoring(hypo,hate_inst)[1]
        elif(metric=='counter_argument'):
            final_score=counter_argument_score.scoring(hypo,hate_inst)[1]
        elif(metric=='toxicity'):
            final_score=toxicity_score.scoring(hypo,hate_inst)[1]
        

        data_dict = {
                     metric:final_score,
                     }
        key = files
        print(key.split('/Results/'))
        scores[key.split('/Results/')[0]] = data_dict
        print(f'Key:{key}')
#         for d in data_dict:
#             print(d+ ':' + str(data_dict[d]))
        print("===============================")
        it +=1
        continue
    
    #### For baseline metric
    hypo = []
    refs = []
    hate_inst=[]
    mp = mappings_baseline(ref_dict,ref_dict)
    for key in ref_dict:
        if key in mp.keys():  
            for sentences in ref_dict[key]['counterspeech']:
                hypo.append(sentences)
                cs_refs = ref_dict[mp[key]]['counterspeech']
                refs.append(cs_refs)
                hate_inst.append(ref_dict[mp[key]]['hatespeech'])
        
    print(len(hypo),len(refs))
    params = [hypo,refs]
    if(metric=='bleurt'):
        final_score=bleurt_score.score(params)[1]
    elif(metric=='argument'):
        final_score=argument_score.scoring(hypo)[1]
    elif(metric=='counterspeech'):
        final_score=counterspeech_score.scoring(hypo)[1]
    elif(metric=='upvote_prob'):
        final_score=dialog_upvote.scoring(hypo,hate_inst)[1]
    elif(metric=='width_prob'):
        final_score=dialog_width.scoring(hypo,hate_inst)[1]
    elif(metric=='depth_prob'):
        final_score=dialog_depth.scoring(hypo,hate_inst)[1]
    elif(metric=='counter_argument'):
            final_score=counter_argument_score.scoring(hypo,hate_inst)[1]
    elif(metric=='toxicity'):
        final_score=toxicity_score.scoring(hypo,hate_inst)[1]

    data_dict = {
                 metric:final_score,
                 }
    scores['Baseline'] = data_dict
    print('Baseline')
#     for d in data_dict:
#         print(d+ ':' + str(data_dict[d]))
    print("===============================")
    it +=1

    
    
    write_in='metrics_eval_utest/Newer_metrics_'+metric+'_on_'+train_sample+".pkl"
    with open(write_in, 'wb') as outfile:
        pkl.dump(scores, outfile,protocol=2)








if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('dataset',
                           metavar='--d',
                           type=str,
                           help='location where the dataset is saved')
    
    my_parser.add_argument('metric',
                           metavar='--m',
                           type=str,
                           help='which metric to be used')
    
    device='cuda'
    if torch.cuda.is_available() and device=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
            deviceID = [0]
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
    metric=args.metric
    
    heavy_metrics(dataset,dataset,metric)
    
    
    