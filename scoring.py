"""
To calculate metrics for generated samples.
"""

import json
import nltk
import math
from collections import Counter
# from rouge_score import rouge_scorer
import argparse
from bert_score import score
from eval_models import *
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util
from distinct_n.metrics import distinct_n_corpus_level
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm import tqdm


def clip_after_last_full_stop(input_string):
    last_full_stop_index = input_string.rfind('.')
    
    if last_full_stop_index != -1:
        clipped_string = input_string[:last_full_stop_index + 1]
        # The +1 is to include the last full stop in the clipped string
        return clipped_string
    else:
        # If there is no full stop, return the original string
        return input_string


def calculate_ngram_entropy(sentences, n):
    total_ngrams = 0
    ngram_counts = Counter()

    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence.lower())  # Tokenize the sentence into words
        ngrams = nltk.ngrams(tokens, n)  # Generate n-grams for the sentence
        ngram_counts.update(ngrams)  # Update n-gram counts
        total_ngrams += len(tokens) - n + 1  # Update total n-grams count

    entropy = 0.0

    for count in ngram_counts.values():
        probability = count / total_ngrams
        entropy -= probability * math.log2(probability)  # Calculate entropy for each n-gram

    return entropy


def calculate_self_bleu(corpus, n):
    self_bleu_scores = []
    for i, hypothesis in tqdm(enumerate(corpus), desc='Self BLEU-'+str(n), total=len(corpus)):
        references = corpus[:i] + corpus[i+1:]  # Exclude current sentence from references
        self_bleu = sentence_bleu(references, hypothesis, weights=[1/n for _ in range(n)])
        self_bleu_scores.append(self_bleu)
    return sum(self_bleu_scores) / len(self_bleu_scores)


def calculate_bleu_2(cs, org_cs):
    cs_tokenized = [sentence.split() for sentence in cs]
    org_cs_tokenized = [sentence.split() for sentence in org_cs]
    return corpus_bleu(org_cs_tokenized, cs_tokenized, weights=(0.5, 0.5))


def calculate_rouge_2(cs, org_cs):
    scorer = rouge_scorer.RougeScorer(['rouge2'])
    scores = scorer.score(org_cs, cs)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metrics scoring script')
    parser.add_argument('--file_name', type=str, help='File name in Generated_Samples folder.')
    args = parser.parse_args()
    
    bleurt_score = Bleurt(model_path="Elron/bleurt-large-512", cache_path='../../Saved_models', max_length=400, batch_size=128, use_gpu=True, gpu='cuda:1')

    argument_score = Argument_scoring(model_path='chkla/roberta-argument', cache_path='../../Saved_models', max_length=400, batch_size=16, use_gpu=True, gpu='cuda:1')

    dialog_upvote = Dialog_upvote_scoring(model_path='microsoft/DialogRPT-updown',cache_path='../../Saved_models', max_length=400, batch_size=16, use_gpu=True, gpu='cuda:1')

    dialog_width = Dialog_upvote_scoring(model_path='microsoft/DialogRPT-width', cache_path='../../Saved_models', max_length=400, batch_size=16, use_gpu=True, gpu='cuda:1')

    dialog_depth = Dialog_upvote_scoring(model_path='microsoft/DialogRPT-depth', cache_path='../../Saved_models', max_length=100, batch_size=16, use_gpu=True, gpu='cuda:2')

    toxicity_score = Toxic_HateXplain_scoring(model_path=None, cache_path='../../Saved_models', max_length=400, batch_size=16, use_gpu=True, gpu='cuda:2')

    old_counterspeech_score = Argument_scoring(model_path='Hate-speech-CNERG/counterspeech-quality-bert', cache_path='../../Saved_models', max_length=400, batch_size=16, use_gpu=True, gpu='cuda:2')
    
    new_counterspeech_score = Argument_scoring(model_path='Hate-speech-CNERG/new-counterspeech-score', cache_path=None, max_length=400, batch_size=16, use_gpu=True, gpu='cuda:2')

    counter_argument_score = Counter_argument_scoring(model_path='Hate-speech-CNERG/argument-quality-bert', cache_path='../../Saved_models', max_length=400, batch_size=8, use_gpu=True, gpu='cuda:0')
    
    div_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1', device='cuda:0')
    gruen_score = Gruen(use_gpu=True, gpu='cuda:0')
    mover_score = MoverScore(use_gpu=True, gpu='cuda:0', n_gram = 1)
    
    cs = []
    ref_cs = []
    hs = []
    ref_hs = []
    hs_cs = []
    
    with open('Generated_Samples/' + args.file_name) as f:
        d = json.load(f)

        samples = d['samples']
        for sample in samples.values():
            cs_ = sample['counterspeech_model']
            hs_ = sample['hatespeech']
            ref_hs_ = sample['org_hate']
            ref_cs_ = sample['org_counter']

            for x in cs_:
                if len(x)>10:
                    hs_cs_ = '<HATESPEECH> ' + hs_ + ' <COUNTERSPEECH> ' + x
                    cs.append(clip_after_last_full_stop(x))
                    hs.append(clip_after_last_full_stop(hs_))
                    ref_hs.append(clip_after_last_full_stop(ref_hs_))
                    ref_cs.append(clip_after_last_full_stop(ref_cs_))
                    hs_cs.append(clip_after_last_full_stop(hs_cs_))

    s1 = argument_score.scoring(cs)
    s2 = dialog_upvote.scoring(cs, hs)
    s3 = dialog_width.scoring(cs, hs)
    s4 = dialog_depth.scoring(cs, hs)
    s5 = toxicity_score.scoring(cs, hs)
    s6 = old_counterspeech_score.scoring(cs)    
    s6_dash = new_counterspeech_score.scoring(hs_cs)

    s7 = counter_argument_score.scoring(cs, hs)
    s8 = score(cs, ref_cs, lang="en", verbose=False)[2].mean().item()
    s9 = bleurt_score.score([cs, ref_cs])
    s10 = avg_novelty(cs, ref_cs)
    bleu, gleu, meteor = nltk_metrics(cs, ref_cs)
    
    embs = div_model.encode(cs)
    cosine_scores = util.cos_sim(embs, embs)
    n = cosine_scores.shape[0]
    total_sim = np.sum(np.array(cosine_scores))

    for i in range(n):
        total_sim -= cosine_scores[i][i]

    if n!=1:
        avg_sim = total_sim/(n*(n-1))
        
    div = 1 - avg_sim.item()
    
    dist1 = distinct_n_corpus_level(cs, 1)   
    dist2 = distinct_n_corpus_level(cs, 2)

    ent1 = calculate_ngram_entropy(cs, 1)
    ent2 = calculate_ngram_entropy(cs, 2)
    
    sb1 = calculate_self_bleu(cs, 1)
    sb2 = calculate_self_bleu(cs, 2)
    gruen = gruen_score.score(cs)
    movr = mover_score.score(cs, ref_cs)
    
    b2 = calculate_bleu_2(cs, ref_cs)
    
    with open('Results/' + args.file_name, 'w') as f:
        json.dump({
            'Argument Score': str(s1),
            'Dialog Upvote': str(s2),
            'Dialog Width': str(s3),
            'Dialog Depth': str(s4),
            'Toxicity': str(s5),
            'Old Counterspeech Score': str(s6),
            'New Counterspeech Score': str(s6_dash),
            'Counter-Argument Score': str(s7),
            'Bert Score': str(s8),
            'Bleurt Score': str(s9),
            'Novelty': str(s10),
            'gleu': str(gleu),
            'bleu': str(bleu),
            'meteor': str(meteor),
            'Diversity': str(div),
            'dist-1': str(dist1),            
            'dist-2': str(dist2),
            'ent-1': str(ent1),            
            'ent-2': str(ent2),
            'sb-1': str(sb1),
            'sb-2': str(sb2),
            'b-2': str(b2),
            'Gruen': str(gruen),
            'Mover': str(movr)

        }, f, indent=4)
     
    print("Result saved")
        
        