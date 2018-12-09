import dill as pickle
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

sys.path.append('sears') # noqa
import replace_rules
import paraphrase_scorer
import onmt_model
import numpy as np

import spacy

nlp = spacy.load('en')
tokenizer = replace_rules.Tokenizer(nlp)
ps = paraphrase_scorer.ParaphraseScorer(gpu_id=0)

#################################################################################################################################
# get data, labels, vals, and vals_labels from training set
# tokenizer.clean_for_model

import json
"""
vals = [..., (paragraph, question), ...]
labels = [..., answer, ...]
"""
def get_data():
    vals = []
    labels = []
    with open('MC_data/squad-dev-v1.1.json', 'r') as squad_data_file:
        squad_data = json.load(squad_data_file)

        for passage in squad_data['data']:

            for paragraph in passage['paragraphs']:

                paragraph_context = paragraph['context']

#                 data_instance = {}
#                 data_instance['paragraph'] = paragraph_context
#                 data_instance['qas'] = []

                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = set([answer_dict['text'] for answer_dict in qa['answers']])

#                     qa_instance = {'question': question, 'answer': answer}
#                     data_instance['qas'].append(qa_instance)

#                     data_instance = {
#                         'paragraph': paragraph_context,
#                         'question': question,
#                         'correct_answer': answer
#                     }
#                     data.append(data_instance)

                    val = (paragraph_context, question)
                    label = answer
                    vals.append(val)
                    labels.append(label)
    return vals, labels
# len(get_data()[0])


#################################################################################################################################
# bidaf model to make MC prediction

import requests

#     payload = [
#         {'passage': 'This is a passage.', 'question': 'What is this?'},
#         {'passage': 'The cat went to the church.', 'question': 'Where did the cat go?'}
#     ]
def mc_batch_predict(vals):
    url_batch = 'http://localhost:8001/predict_batch'
    data = [{'passage': val[0], 'question': val[1]} for val in vals]
    
    size = 50
    splitted_data = [data[i:i+size] for i in range(0, len(data), size)]
    
    result = []
    for chunk in splitted_data:
        try:
            response = requests.post(url_batch, json=chunk).json()
            #result.extend([instance['answer'] for instance in response])
            result.extend(response)
            #time.sleep(.100)
        except Exception as e:
            # print(chunk)
            #print(response)
            raise RuntimeError()
        
    return result

from queue import Queue
from threading import Thread
def mc_batch_predict_multithreading(vals):

    def query(url, data, level=3):
        try:
            response = requests.post(url, json=data).json()
            #result.extend([instance['answer'] for instance in response])
            return response
        except Exception as e:
            if level == 0:
                raise e
            data1 = data[:len(data)//2]
            data2 = data[len(data)//2:]
            res1 = query(url, data1, level-1)
            res2 = query(url, data2, level-1)
            res1.extend(res2)
            return res1

    def query_helper(q, url, result):
        while not q.empty():
            work = q.get()
            response = query(url, work[1])
            #result.extend([instance['answer'] for instance in response])
            result[work[0]] = response

            q.task_done()
        return True

    #url_batch = 'http://localhost:8000/predict/batch/machine-comprehension-hard'
    url_batch_0 = 'http://localhost:8001/predict_batch'
    url_batch_1 = 'http://localhost:8002/predict_batch'
    url_batch_2 = 'http://localhost:8003/predict_batch'
    url_batch_3 = 'http://localhost:8004/predict_batch'
    urls_batch = [url_batch_0, url_batch_1, url_batch_2, url_batch_3]

    data = [{'passage': val[0], 'question': val[1]} for val in vals]
    size = 64
    splitted_data = [data[i:i+size] for i in range(0, len(data), size)]
    q = Queue()
    for i, chunk in enumerate(splitted_data):
        q.put((i, chunk))

    num_worker = min(len(urls_batch), len(splitted_data))

    result = [{} for chunk in splitted_data]

    threads = []
    for i in range(num_worker):
        worker = Thread(target=query_helper, args=(q, urls_batch[i], result))
        worker.daemon = True
        worker.start()
        threads.append(worker)

    q.join()

    for worker in threads:
        worker.join()

    result = [answer for response in result for answer in response]
    return result

def mc_single_predict(vals):
    url_single = 'http://localhost:8000/predict/machine-comprehension-hard'
    data = [{'passage': val[0], 'question': val[1]} for val in vals]
    
    result = []
    #for chunk in splitted_data:
    for pair in data:
        try:
            response = requests.post(url_single, json=pair).json()
            #result.append([instance['answer'] for instance in response])
            result.append(response['best_span_str'])
        except Exception:
            # print(chunk)
            raise RuntimeError()
        
    return result


#################################################################################################################################
# predict, pick out correct predicted questions

def is_adversarial(prediction, label):
    # return prediction != label
    def overlap(answer, expected):
        return expected in answer and len(prediction) < 2 * len(label)

    return prediction not in label

def filter_data(vals, labels, keep_adversarial):
    predictions = mc_batch_predict_multithreading(vals)
#     right = [data_instance for data_instance in prediction if is_adversarial(data_instance) == keep_adversarial]
    right = [i for i in range(len(vals)) if is_adversarial(predictions[i], labels[i]) == keep_adversarial]
    right_vals = [vals[i] for i in right]
    right_preds = [labels[i] for i in right]
    return right, right_vals, right_preds


#################################################################################################################################
# compute SEAs using real time paraphrase function
import collections
from copy import copy

# paraphrase question through translation
# paraphrase('The plane was late and the detectives were waiting at the airport all morning.')
def paraphrase(question, topk=10, threshold=-10, ):
    question_for_onmt = onmt_model.clean_text(' '.join([x.text for x in nlp.tokenizer(question)]), only_upper=False)
    paraphrases = ps.generate_paraphrases(question_for_onmt, topk=topk, edit_distance_cutoff=4, threshold=threshold)
    texts = tokenizer.clean_for_model(tokenizer.clean_for_humans([x[0] for x in paraphrases]))
    # texts is list of paraphrased question
    return texts, paraphrases

def find_adversarials(val, label, topk=10, threshold=-10):
    # origin_pred = mc_batch_predict([val])[0]
    
    texts, paraphrases = paraphrase(val[1], topk, threshold)
    
    paraphrased_val = [(val[0], new_question) for new_question in texts if new_question != '']
    
    paraphrased_pred = mc_batch_predict_multithreading(paraphrased_val)
#     fs = [(texts[i], paraphrases[i][1]) for i in np.where(preds != orig_pred)[0]] # TODO: implement adversarial definition
#     fs = [paraphrased_pred for paraphrased_pred in paraphrased_preds if is_adversarial(paraphrased_pred)]
    fs = [para_val for i, para_val in enumerate(paraphrased_val) if is_adversarial(paraphrased_pred[i], label)]
    return fs


def generate_adversarials_for_all_data(right_vals, right_preds):
    flips = collections.defaultdict(lambda: [])
    print(len(right_vals))
    for index in range(len(right_vals)):
        val = right_vals[index]
        label = right_preds[index]
        if index % 1000 == 0:
            print(index)
#         if val[idx] in flips:
#             continue
        fs = find_adversarials(val, label, topk=100, threshold=-10)
        flips[index].extend(fs)
    return flips

# flips = generate_adversarials_for_all_data(right_vals[40:])

#################################################################################################################################
# extract rules from SEA
import replace_rules

def extract_rules(right_vals, flips):
    original_questions = [val[1] for val in right_vals]
    tr2 = replace_rules.TextToReplaceRules(nlp, original_questions, [], min_freq=0.005, min_flip=0.00, ngram_size=4)
    
    frequent_rules = []
    rule_idx = {}
#     rule_flips = {}
    for z, index in enumerate(flips):
        original_question = right_vals[index][1]
        paraphrased_question = [para_val[1] for para_val in flips[index]]
        rules = tr2.compute_rules(original_question, paraphrased_question, use_pos=True, use_tags=True)
        for rs in rules:
            for r in rs:
                if r.hash() not in rule_idx:
                    i = len(rule_idx)
                    rule_idx[r.hash()] = i
#                     rule_flips[i] = []
                    frequent_rules.append(r)
                i = rule_idx[r.hash()]
#                 rule_flips[i].append(z)
        if z % 500 == 0:
            print (z)
    return frequent_rules, tr2

#################################################################################################################################
# run those two steps coverage test to select final SEARs from the rules
# potential problem: in scoring process?
from rule_picking import disqualify_rules, choose_rules_coverage

def chunks(all_data, chunk_size):
    for i in range(0, len(all_data), chunk_size):
        yield all_data[i: i+chunk_size]

def get_model_preds(right_vals, right_preds, frequent_rules, tr2):
    # modified tokenizer.tokenize mc_val_adhoc, token_right needs to be list of (paragraph, question_tokens)
    token_right = tokenizer.tokenize(right_vals, mc_adhoc=True)
    model_preds = {}


    to_compute_all = set()
    for i, r in enumerate(frequent_rules):
        idxs = list(tr2.get_rule_idxs(r))
        to_apply = [token_right[x] for x in idxs]
        # modified r.apply_to_texts mc_val_adhoc, nt needs to list of tuples (paragraph, question)
        applies, nt = r.apply_to_texts(to_apply, fix_apostrophe=False, mc_adhoc=True)
        to_compute = [x for x in nt if x not in model_preds]
        to_compute_all.update(to_compute)
        if len(to_compute_all) >= 10000 or i == len(frequent_rules) - 1:
            print('querying allennlp on frequent rule number %s roughly at %s with size %s' % (i, '%.2f'%(i/len(frequent_rules)), len(to_compute_all)))
            preds_all = mc_batch_predict_multithreading(to_compute_all)
            for x, y in zip(to_compute_all, preds_all):
                model_preds[x] = y
            to_compute_all = set()
    return token_right, model_preds

def probably_pick_out_SEARs(right_vals, right_preds, frequent_rules, tr2, model_preds, token_right):
    a = time.time()

    rule_flips = {}
    rule_other_texts = {}
    rule_other_flips = {}
    rule_applies = {}
    print(len(frequent_rules))
    for i, r in enumerate(frequent_rules):
        idxs = list(tr2.get_rule_idxs(r))
        to_apply = [token_right[x] for x in idxs]
        # modified r.apply_to_texts mc_val_adhoc, nt needs to list of tuples (paragraph, question)
        applies, nt = r.apply_to_texts(to_apply, fix_apostrophe=False, mc_adhoc=True)
        applies = [idxs[x] for x in applies] # [ indices of right_vals that r can be applied to ]
        old_texts = [right_vals[index] for index in applies]
#         old_labels = right_preds[applies]
        old_labels = [right_preds[index] for index in applies]

        new_labels = np.array([model_preds[x] for x in nt])
#         where_flipped = np.where(new_labels != old_labels)[0]
        where_flipped = [index for index in range(len(new_labels)) if is_adversarial(new_labels[index], old_labels[index])]
        flips = sorted([applies[x] for x in where_flipped])
        rule_flips[i] = flips
        rule_other_texts[i] = nt
        rule_other_flips[i] = where_flipped
        rule_applies[i] = applies

        if i % 5000 == 0:
            print(i)
    print(time.time() - a)
    
    # TODO
    really_frequent_rules = [i for i in range(len(rule_flips)) if len(rule_flips[i]) > 35]
    threshold = -7.15
    orig_scores = {}
    for i, t in enumerate(right_vals):
        # modified ps.score_sentences
        orig_scores[i] = ps.score_sentences(t, [t], mc_adhoc=True)[0]
        
    ps_scores = {}
    ps.last = None
    rule_scores = []
    rejected = set()
    print(len(really_frequent_rules))
    for idx, i in enumerate(really_frequent_rules):
        orig_texts =  [right_vals[z] for z in rule_applies[i]]
        orig_scor = [orig_scores[z] for z in rule_applies[i]]
        scores = np.ones(len(orig_texts)) * -50
    #     if idx in rejected:
    #         rule_scores.append(scores)
    #         continue
        decile = np.ceil(.1 * len(orig_texts))
        new_texts = rule_other_texts[i]
        bad_scores = 0
        for j, (o, n, orig) in enumerate(zip(orig_texts, new_texts, orig_scor)):
            if o not in ps_scores:
                ps_scores[o] = {}
            if n not in ps_scores[o]:
                if n == '':
                    score = -40
                else:
                    # modified ps.score_sentences
                    score = ps.score_sentences(o, [n], mc_adhoc=True)[0]
                ps_scores[o][n] = min(0, score - orig)
            scores[j] = ps_scores[o][n]
            if ps_scores[o][n] < threshold:
                bad_scores += 1
            if bad_scores >= decile:
                rejected.add(idx)
                break
        rule_scores.append(scores)

        if idx % 1000 == 0:
            print(idx)
            
    rule_flip_scores = [rule_scores[i][rule_other_flips[really_frequent_rules[i]]] for i in range(len(rule_scores))]
    frequent_flips = [np.array(rule_applies[i])[rule_other_flips[i]] for i in really_frequent_rules]
    rule_precsupports = [len(rule_applies[i]) for i in really_frequent_rules]
    
    threshold=-7.15
    # x = choose_rules_coverage(fake_scores, frequent_flips, frequent_supports,
    disqualified = disqualify_rules(rule_scores, frequent_flips,
                              rule_precsupports, 
                          min_precision=0.0, min_flips=6, 
                             min_bad_score=threshold, max_bad_proportion=.10,
                              max_bad_sum=999999)
    
    threshold=-7.15
    a = time.time()
    x = choose_rules_coverage(rule_flip_scores, frequent_flips, None,
                              None, len(right_preds),
                                    frequent_scores_on_all=None, k=10, metric='max',
                          min_precision=0.0, min_flips=0, exp=True,
                             min_bad_score=threshold, max_bad_proportion=.1,
                              max_bad_sum=999999,
                             disqualified=disqualified,
                             start_from=[])
    print(time.time() -a)
    support_denominator = float(len(right_preds))
    soup = lambda x: len(rule_applies[really_frequent_rules[x]]) / support_denominator 
    prec = lambda x: frequent_flips[x].shape[0] / float(len(rule_scores[x]))
    fl = len(set([a for r in x for a in frequent_flips[r]]))
    print('Instances flipped: %d (%.2f)' % (fl, fl / float(len(right_preds))))
    print('\n'.join(['%-5d %-5d %-5d %-35s f:%d avg_s:%.2f bad_s:%.2f bad_sum:%d Prec:%.2f Supp:%.2f' % (
                    i, x[i], really_frequent_rules[r],
                    frequent_rules[really_frequent_rules[r]].hash().replace('text_', '').replace('pos_', '').replace('tag_', ''),
                    frequent_flips[r].shape[0],
                    np.exp(rule_flip_scores[r]).mean(), (rule_scores[r] < threshold).mean(),
                    (rule_scores[r] < threshold).sum(), prec(r), soup(r)) for i, r in enumerate(x)]))
    
    return x


#################################################################################################################################
def main():
    all_vals, all_labels = get_data()
    print('length of all_vals:')
    print(len(all_vals))

    total_size = 10570
    #test_size = 20
    test_size = total_size
    vals = all_vals[:test_size]
    labels = all_labels[:test_size]

    # vals = all_vals
    # labels = all_labels

    print('filtering data:')
    right, right_vals, right_preds = filter_data(vals, labels, keep_adversarial=False)
    print(len(right))

    print('pickling filtered data...')
    with open('cache/right_data.pkl', 'wb') as f:
        pickle.dump(right, f)
        pickle.dump(right_vals, f)
        pickle.dump(right_preds, f)

    print('generating advesarials for all data:')
    flips = generate_adversarials_for_all_data(right_vals, right_preds)

    print('pickling flips...')
    with open('cache/flips.pkl', 'wb') as f:
        pickle.dump(flips, f)

    print('extracting rules:')
    # len(flips[0])
    frequent_rules, tr2 = extract_rules(right_vals, flips)
    print('num frequent_rules: %s' % len(frequent_rules))

    print('pickling frequent_rules...')
    with open('cache/frequent_rules.pkl', 'wb') as f:
        pickle.dump(frequent_rules, f)

    print('getting model preds...')
    token_right, model_preds = get_model_preds(right_vals, right_preds, frequent_rules, tr2)

    print('pickling model preds...')
    with open('cache/model_preds.pkl', 'wb') as f:
        pickle.dump(model_preds, f)

    print('pickling token right...')
    with open('cache/token_right.pkl', 'wb') as f:
        pickle.dump(token_right, f)

    print('probably picking out SEARs:')
    sears = probably_pick_out_SEARs(right_vals, right_preds, frequent_rules, tr2, model_preds, token_right)

    print(len(sears))

    with open('sears_MC.pkl', 'wb') as f:
        pickle.dump(sears, f)


def resume():
    print('loading filtered data...')
    with open('cache/right_data.pkl', 'rb') as f:
        right = pickle.load(f)
        right_vals = pickle.load(f)
        right_preds = pickle.load(f)

    print('loading flips...')
    with open('cache/flips.pkl', 'rb') as f:
        flips = pickle.load(f)

    print('extracting rules:')
    # len(flips[0])
    frequent_rules, tr2 = extract_rules(right_vals, flips)
    print('num frequent_rules: %s' % len(frequent_rules))

    print('pickling frequent_rules...')
    with open('cache/frequent_rules.pkl', 'wb') as f:
        pickle.dump(frequent_rules, f)

    print('getting model preds...')
    token_right, model_preds = get_model_preds(right_vals, right_preds, frequent_rules, tr2)

    print('pickling model preds...')
    with open('cache/model_preds.pkl', 'wb') as f:
        pickle.dump(model_preds, f)

    print('pickling token right...')
    with open('cache/token_right.pkl', 'wb') as f:
        pickle.dump(token_right, f)

    print('probably picking out SEARs:')
    sears = probably_pick_out_SEARs(right_vals, right_preds, frequent_rules, tr2, model_preds, token_right)

    print(len(sears))

    with open('sears_MC.pkl', 'wb') as f:
        pickle.dump(sears, f)


if __name__ == '__main__':
    start_time = time.time()
    #main()
    resume()
    print('total running time: %ss' % (time.time() - start_time))
