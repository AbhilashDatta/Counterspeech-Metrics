import glob
import os
import json
import pandas as pd 
from Generation.eval import *
from transformers import AutoTokenizer,AutoModelForCausalLM
import argparse

path_datasets = 'Datasets/'
path_result   = '../../../HULK_new/Counterspeech/Results/'
save_path     = 'Results/'

#### load tokenizer model
model_path="bert-base-uncased"
cache_path="../../Saved_models/"
tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path,fast=False)


def unnest(df, col, reset_index=False):
    col_flat = pd.DataFrame([[i, x] 
                       for i, y in df[col].apply(list).iteritems() 
                           for x in y], columns=['I', col])
    col_flat = col_flat.set_index('I')
    df = df.drop(col, 1)
    df = df.merge(col_flat, left_index=True, right_index=True)
    if reset_index:
        df = df.reset_index(drop=True)
    return df


def mappings(gen_dict,ref):
    mp = {}
    for key in ref.keys():
        ref_ht = ref[key]["hatespeech"]
        for key2 in gen_dict['samples'].keys():
            gen_ht = gen_dict['samples'][key2]["org_hate"]
            if gen_ht==ref_ht:
                mp[key2]   = key
    return mp

def metrics(train_sample,gen_sample,baseline):
    train_path = glob.glob(path_datasets+'/'+gen_sample+'/Train.json')[0]
    print(train_path)
#     gen        = glob.glob(save_path+'/**'+gen_sample+'_huggingface_base.json')
#     gen         = glob.glob(save_path+'/**'+gen_sample+'_huggingface_base.json')
#     gen2        = glob.glob(save_path+'/**'+gen_sample+'_topic_argument_huggingface_base.json')
#     gen3        = glob.glob(save_path+'/**'+gen_sample+'*'+'_sz_3_'+'*'+'.json')
    gen4        = glob.glob(save_path+'/'+'*'+gen_sample+'*'+'.json')
#     gen5        = glob.glob('Results/'+'Generator-DialoGPT-small_on_'+gen_sample+'_huggingface_base.json')
#     gen=gen+gen2+gen3
    baseline_a   =  glob.glob('Results/'+'Generator-DialoGPT-small_on_*'+gen_sample+'_huggingface_base.json')
    baseline_b   =  glob.glob('Results/'+'Generator-'+gen_sample+'__DialoGPT-small_on_'+gen_sample+'_huggingface_base.json')
#
    gen = gen4
    if baseline==True:
        gen = baseline_a + baseline_b
#     gen = gen3

    print('first',train_sample,'second',gen_sample)
                            
    gen = sorted(gen,key = os.path.getmtime)
    print(gen)
    print(path_datasets+'/'+gen_sample+'/Test.json')
    ref        = glob.glob(path_datasets+'/'+gen_sample+'/Test.json')[0]
    ref_csv  = pd.read_json(ref)
    ref_dict={}
    for index,row in ref_csv.iterrows():
        ref_dict[index]={'hatespeech':row['initiator_message'],'counterspeech':row['reply_message']}
           
    for item in gen:
        print(item) 
    it = 1
    scores={}
    
    

    write_in='Metrics_eval_results/Older_metrics_on_'+gen_sample+'_'+train_sample+".json"
    
    #uncomment while runnig_baselines
    if baseline==True:
        write_in='Metrics_eval_results/Older_metrics_on_'+gen_sample+'_'+'baseline.json'
    
    try:
        with open(write_in, 'r') as outfile:
             scores=json.load(outfile)
    except FileNotFoundError:
        pass
    
    model_list = [ 'Argme_pro_con_DialoGPT-small', 
               'arguna__token_DialoGPT-small',
               'CreateDebate_pro_con_DialoGPT-small',
               'winning_arg__token_DialoGPT-small'
             ] 
    for files in gen:
        
        
#         try:
#             temp=scores[files]
#             bk = False
#             for model_name in model_list:
#                 if model_name in model_list:
#                     bk = True
#             if bk==False:
#                 continue
#             continue
#         except KeyError:
#             pass
        
        with open(files, 'r') as file:
            gen_dict  = json.loads(file.read())
            
        
        try:
            temp=scores[files.split('/Results/')[0]]
            continue
        except KeyError:
            pass
        
        hypo = []
        refs = []
        
#         print(type(ref_dict['samples']))
        mp = mappings(gen_dict,ref_dict)
        print(len(mp))
        for key in gen_dict['samples']:
            if key in mp.keys():  
                for sentences in gen_dict['samples'][key]['counterspeech_model']:
                    hypo.append(sentences)
                    cs_refs = ref_dict[mp[key]]['counterspeech']
                    refs.append(cs_refs)
    
        train = pd.read_json(train_path)
        train = unnest(train,'reply_message',reset_index=True)
        train_set = list(zip(train['initiator_message'].tolist(),train['reply_message'].tolist()))
        params = [hypo,refs]

        bleu, gleu, meteor_ = nltk_metrics(params,tokenizer)
        train_corpus = training_corpus(train_set)
        diversity, novelty = diversity_and_novelty(train_corpus,hypo)
        data_dict = {
                     'gleu':gleu,
                     'bleu':bleu,
                     'diversity':diversity,
                     'novelty':novelty, 
                     'meteor':meteor_
                   }
        key = files
        print(key.split('/Results/'))
        scores[key.split('/Results/')[0]] = data_dict
        print(f'Key:{key}')
        for d in data_dict:
            print(d+ ':' + str(data_dict[d]))
        print("===============================")
        it +=1 

    
    write_in='Metrics_eval_results/Older_metrics_on_'+gen_sample+'_'+train_sample+".json"

        #uncomment while runnig_baselines
    if baseline==True:
        write_in='Metrics_eval_results/Older_metrics_on_'+gen_sample+'_'+'baseline.json'

    with open(write_in, 'w') as outfile:
         json.dump(scores, outfile,indent=4)








if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('argument',
                           metavar='--d',
                           type=str,
                           help='location where the argument dataset is saved')
    
    my_parser.add_argument('dataset',
                           metavar='--d',
                           type=str,
                           help='location where the dataset is saved')
    
    baseline = False  
    #switch this on while running on baselines.
    
    args = my_parser.parse_args()
    dataset=args.dataset
    argumnet_dataset=args.argument
    metrics(argumnet_dataset,dataset,baseline)
    
    
    