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

def heavy_metrics(train_sample,gen_sample,metric,baseline):
#     gen1        = glob.glob(save_path+'/**'+gen_sample+'_huggingface_base.json')
#     gen2        = glob.glob(save_path+'/**'+gen_sample+'_topic_argument_huggingface_base.json')
#     gen3        = glob.glob(save_path+'/**'+gen_sample+'_implications_huggingface_base.json')
#    gen4        = glob.glob(save_path+'/**'+gen_sample+'*'+'_sz_3_'+'*'+'.json')
    gen5         =  glob.glob(save_path+'/'+'*'+gen_sample+'*_huggingface_base.json')
    baseline_a   =  glob.glob('Results/'+'Generator-DialoGPT-small_on_*'+gen_sample+'_huggingface_base.json')
    baseline_b   =  glob.glob('Results/'+'Generator-'+gen_sample+'__DialoGPT-small_on_'+gen_sample+'_huggingface_base.json')
#     Generator-Gab__DialoGPT-small_on_Gab_huggingface_base.json
#     Generator-DialoGPT-small_on_CONAN_MT_New_huggingface_base.json
    gen = gen5
    
    if baseline==True:
        gen = baseline_a + baseline_b
#     Generator-DialoGPT-small_on_Reddit_huggingface_base.json
    
    gen = sorted(gen,key = os.path.getmtime)
    
    print(save_path+'/*'+train_sample+'*'+gen_sample+'_huggingface_base.json')
    
    print(path_datasets+'/'+gen_sample+'/Test.json')
    ref        = glob.glob(path_datasets+'/'+gen_sample+'/Test.json')[0]
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
        bleurt_score=Bleurt(model_path="Elron/bleurt-large-512",cache_path='../../Saved_models',max_length=400, batch_size=16,use_gpu=True)
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
#         toxicity_score=Toxicity_model(max_length=100, batch_size=8,use_gpu=True)

        
    
    
    scores={}
    write_in='Metrics_eval_results/Newer_metrics_'+metric+'_on_'+gen_sample+"_"+gen_sample+".json"
        
        
    #uncomment while runnig_baselines
    if baseline==True:
        write_in='Metrics_eval_results/Newer_metrics_'+metric+'_on_'+gen_sample+'_'+'baseline.json'
    try:
        with open(write_in, 'r') as outfile:
             scores=json.load(outfile)
    except FileNotFoundError:
        pass
    print(scores)
    for files in gen:
        with open(files, 'r') as file:
            gen_dict  = json.loads(file.read())
        
        try:
            temp=scores[files]
            continue
        except KeyError:
            pass
        
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
            final_score=bleurt_score.score(params)
        elif(metric=='argument'):
            final_score=argument_score.scoring(hypo)
        elif(metric=='counterspeech'):
            final_score=counterspeech_score.scoring(hypo)
        elif(metric=='upvote_prob'):
            final_score=dialog_upvote.scoring(hypo,hate_inst)
        elif(metric=='width_prob'):
            final_score=dialog_width.scoring(hypo,hate_inst)
        elif(metric=='depth_prob'):
            final_score=dialog_depth.scoring(hypo,hate_inst)
        elif(metric=='counter_argument'):
            final_score=counter_argument_score.scoring(hypo,hate_inst)
        elif(metric=='toxicity'):
            final_score=toxicity_score.scoring(hypo,hate_inst)
            
            
        data_dict = {
                     metric:str(final_score),
                     }
        key = files
        print(key.split('/Results/'))
        scores[key.split('/Results/')[0]] = data_dict
        print(f'Key:{key}')
        for d in data_dict:
            print(d+ ':' + str(data_dict[d]))
        print("===============================")
        it +=1 
    
    #### For baseline metric
    try:
        temp=scores['Baseline']
    except:
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
            final_score=bleurt_score.score(params)
        elif(metric=='argument'):
            final_score=argument_score.scoring(hypo)
        elif(metric=='counterspeech'):
            final_score=counterspeech_score.scoring(hypo)
        elif(metric=='upvote_prob'):
            final_score=dialog_upvote.scoring(hypo,hate_inst)
        elif(metric=='width_prob'):
            final_score=dialog_width.scoring(hypo,hate_inst)
        elif(metric=='depth_prob'):
            final_score=dialog_depth.scoring(hypo,hate_inst)
        elif(metric=='counter_argument'):
            final_score=counter_argument_score.scoring(hypo,hate_inst)
        elif(metric=='toxicity'):
            final_score=toxicity_score.scoring(hypo,hate_inst)


        data_dict = {
                     metric:str(final_score),
                     }
        scores['Baseline'] = data_dict
        print(scores.keys())
        for d in data_dict:
            print(d+ ':' + str(data_dict[d]))
        print("===============================")
        it +=1

    
    
    write_in='Metrics_eval_results/Newer_metrics_'+metric+'_on_'+gen_sample+"_"+gen_sample+".json"
    #write_in='eval_results_200/Newer_metrics_'+metric+'_on_'+gen_sample+"_"+train_sample+".json"
    
    #uncomment while runnig_baselines
    if baseline==True:
        write_in='Metrics_eval_results/Newer_metrics_'+metric+'_on_'+gen_sample+'_'+'baseline.json'
    
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
#             torch.cuda.set_device(1)
            ##### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
            torch.cuda.set_device(deviceID[0])

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    args = my_parser.parse_args()
    argument_dataset=args.argument
    dataset=args.dataset
    metric=args.metric
    
    
    #switch this signal while running baselines
    baseline = False
    
    heavy_metrics(argument_dataset,dataset,metric,baseline)
    
    
    