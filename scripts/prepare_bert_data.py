import re
import sys
import json
from os.path import join, exists
from transformers import BertTokenizer

def tokenize_no_unk(tokenizer, text):
    split_tokens = []
    for token in tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens):
        wp_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        if wp_tokens == [tokenizer.unk_token]:
            split_tokens.append(token)
        else:
            split_tokens.extend(wp_tokens)
    return split_tokens

def find_sublist(a, b, order=-1):
    if not b: 
        return -1
    counter = 0
    for i in range(len(a)-len(b)+1):
        if a[i:i+len(b)] == b:
            counter += 1
            if counter > order:
                return i
    return -1

#### CAUTIOUS! Precise instrument ####
## General ##
num_eng = re.compile('(\d+(,\d+)*(\.){0,1}\d*)([a-z]+)')

def blank_num_eng(passage):
    try:
        for group in num_eng.findall(passage):
            num, eng = group[0], group[-1]
            target = num + eng
            former, latter = passage.split(target)
            passage = '%s%s %s%s' % (former, num, eng, latter)
        return passage
    except:
        return passage

def remove_substr(sets):
        i = 0
        while i < len(sets):
            substr = False
            for e in sets:
                if sets[i][-1] != e[-1] and sets[i][-1] in e[-1] and \
                   sets[i][0] in range(e[0], e[1]):
                    sets.pop(i)
                    substr = True
                    break
            if not substr:
                i += 1
        return sets

## Date-duration ##
date_dur_par = ('^(((\d|零|一|二|三|四|五|六|七|八|九|十)+|前|去|今|明|後)(周年|年|歲){0,1}){0,1}'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(周){0,1}){0,1}'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(個){0,1}(月){0,1}){0,1}'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(日|號|天){0,1}){0,1}$')
date_dur_full = ('^(((\d|零|一|二|三|四|五|六|七|八|九|十)+|前|去|今|明|後)(年|歲)){0,1}'
                 '((\d|零|一|二|三|四|五|六|七|八|九|十)+(個){0,1}月){0,1}'
                 '((\d|零|一|二|三|四|五|六|七|八|九|十)+周年){0,1}'
                 '((\d|零|一|二|三|四|五|六|七|八|九|十)+周){0,1}'
                 '((\d|零|一|二|三|四|五|六|七|八|九|十)+(日|號|天)){0,1}$')
date_pattern = ('(((\d|零|一|二|三|四|五|六|七|八|九|十)+|前|去|今|明|後)年'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+月){0,1}'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(日|號)){0,1}|'
                '(\d|零|一|二|三|四|五|六|七|八|九|十)+月'
                '((\d|零|一|二|三|四|五|六|七|八|九|十)+(日|號)){0,1}|'
                '(\d|零|一|二|三|四|五|六|七|八|九|十)+(日|號))')
dur_pattern = ('((\d|零|一|二|三|四|五|六|七|八|九|十)+(周年|年|歲)'
               '|(\d|零|一|二|三|四|五|六|七|八|九|十)+個月'
               '|(\d|零|一|二|三|四|五|六|七|八|九|十)+周'
               '|(\d|零|一|二|三|四|五|六|七|八|九|十)+(日|天))')
date_dur_par_re = re.compile(date_dur_par)
date_dur_full_re = re.compile(date_dur_full)
date_re = re.compile(date_pattern)
dur_re = re.compile(dur_pattern)

def find_all_range(tokenizer, par_re, full_re, tokens):
    assert '' not in tokens
    cursor = 0
    cand_tokens = []
    found_strings = []
    for i, token in enumerate(tokens):
        cand_string = tokenizer.convert_tokens_to_string(cand_tokens + [token]).replace(' ', '')
        match = par_re.search(cand_string)
        if not match:
            if cand_tokens:
                cand_string = tokenizer.convert_tokens_to_string(cand_tokens).replace(' ', '')
                full_match = full_re.search(cand_string)
                if full_match:
                    found_strings.append((cursor, cursor+len(cand_tokens), full_match.group()))
                cand_tokens = []
                if par_re.match(token):
                    cand_tokens.append(token)
                    cursor = i
                else:
                    cursor = i + 1
            else:
                cursor = i + 1
        else:
            cand_tokens.append(token)
    if cand_tokens:
        cand_string = tokenizer.convert_tokens_to_string(cand_tokens).replace(' ', '')
        full_match = full_re.search(cand_string)
        if full_match:
            found_strings.append((cursor, cursor+len(cand_tokens), full_match.group()))
    return found_strings

def find_all_date_durs_range(tokenizer, date_dur_par_re, date_dur_full_re, p_tokens):
    all_date_durs = find_all_range(tokenizer, date_dur_par_re, date_dur_full_re, p_tokens)
    all_dates, all_durs = [], []
    for ind, ind_end, cand in all_date_durs:
        date_match = date_re.search(cand)
        if date_match and date_match.group():
            all_dates.append((ind, ind_end, date_match.group()))
        dur_match = dur_re.search(cand)
        if dur_match and dur_match.group():
            all_durs.append((ind, ind_end, dur_match.group()))
    return list(set(all_dates) | set(all_durs))

def find_datedurs(tokenizer, input_tokens_no_unk):
    datedurs = find_all_date_durs_range(tokenizer, date_dur_par_re, date_dur_full_re, input_tokens_no_unk)
    if len(datedurs) > 0:
        datedurs = remove_substr(datedurs)
    return datedurs

## Arithmetic ##
num_par_re = re.compile('^(((零|一|兩|二|三|四|五|六|七|八|九|十|百|千)*){0,1}|\d+(,\d*)*(\.\d*){0,1})$')
num_full_re = re.compile('^(((零|一|兩|二|三|四|五|六|七|八|九|十|百|千)+)|\d+(,\d+)*(\.\d+){0,1})$')

def find_all_num_range(tokenizer, par_re, full_re, tokens):
    assert '' not in tokens
    cursor = 0
    cand_tokens = []
    found_strings = []
    for i, token in enumerate(tokens):
        cand_string = tokenizer.convert_tokens_to_string(cand_tokens + [token]).replace(' ', '')
        match = par_re.search(cand_string)
        if not match:
            if cand_tokens:
                cand_string = tokenizer.convert_tokens_to_string(cand_tokens).replace(' ', '')
                full_match = full_re.search(cand_string)
                if full_match:
                    found_strings.append((cursor, cursor+len(cand_tokens), full_match.group()))
                cand_tokens = []
                if par_re.match(token):
                    cand_tokens.append(token)
                    cursor = i
                else:
                    cursor = i + 1
            else:
                cursor = i + 1
        else:
            cand_tokens.append(token)
    if cand_tokens:
        cand_string = tokenizer.convert_tokens_to_string(cand_tokens).replace(' ', '')
        full_match = full_re.search(cand_string)
        if full_match:
            found_strings.append((cursor, cursor+len(cand_tokens), full_match.group()))
    
    # Post-process
    strings_with_units = []
    for elem in found_strings:
        try:
            start_ind, end_ind, num = elem

            # Unit completement
            if tokens[end_ind] == '公' or tokens[end_ind] == '英':
                unit = ''.join(tokenizer.convert_tokens_to_string(tokens[end_ind:end_ind+2]).split())
                end_ind += 2
            else:
                unit = tokens[end_ind].replace('#', '')
                if unit not in '，。！？；：、（）.,-:;!?':
                    end_ind += 1
            if not unit or unit == tokenizer.sep_token:
                continue

            span = num + unit
            strings_with_units.append((start_ind, end_ind, span))
        except:
            pass
    return strings_with_units

def find_nums(tokenizer, input_tokens_no_unk):
    nums = find_all_num_range(tokenizer, num_par_re, num_full_re, input_tokens_no_unk)
    if len(nums) > 0:
        nums = remove_substr(nums)
    return nums

######################################


if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print('Usage: python3 prepare_bert_data.py <pretrained_model> <split> <dataset_1> <dataset_2> ... <dataset_n>')
        exit(1)

    model_path = sys.argv[1]
    split = sys.argv[2]
    datasets = sys.argv[3:]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    for dataset in datasets:
        data = json.load(open('dataset/%s.json' % dataset))
        passage_count = len(data)
        impossible_questions = 0
        for i, PQA in enumerate(data, start=1):

            # Passage
            raw_passage = PQA['DTEXT'].strip()
            raw_passage = blank_num_eng(raw_passage)
            passage = tokenizer.tokenize(raw_passage)
            passage_no_unk = tokenize_no_unk(tokenizer, raw_passage)
            PID = PQA['DID']
            
            # Embeddings in passage
            datedur_mask = ['0' for token in passage_no_unk]
            datedurs = find_datedurs(tokenizer, passage_no_unk)
            for datedur in datedurs:
                start, end, span = datedur
                for cur in range(start, end):
                    datedur_mask[cur] = '1'
                
            num_mask = ['0' for token in passage_no_unk]
            nums = find_nums(tokenizer, passage_no_unk)
            for num in nums:
                start, end, span = num
                for cur in range(start, end):
                    num_mask[cur] = '1'
                
            # QA pairs
            QAs = []
            for QA in PQA['QUESTIONS']:
                if QA['AMODE'] != 'Single-Span-Extraction' and \
                   'Single-Span-Extraction' not in QA['AMODE'] or \
                   'ANSWER' not in QA:
                    continue
 
                processed_QA = {}
                raw_question = QA['QTEXT'].strip()
                question = tokenizer.tokenize(raw_question)
                question_no_unk = tokenize_no_unk(tokenizer, raw_question)

                raw_answers = [A['ATEXT'].strip() for A in QA['ANSWER']]
                raw_answer_start = QA['ANSWER'][0]['ATOKEN'][0]['start']
                found_answer_starts = [m.start() for m in re.finditer(raw_answers[0], raw_passage)]
                answer_order, best_dist = -1, 10000
                for order, found_start in enumerate(found_answer_starts):
                    dist = abs(found_start - raw_answer_start)
                    if dist < best_dist:
                        best_dist = dist
                        answer_order = order

                answer_no_unk = tokenize_no_unk(tokenizer, raw_answers[0])
                answer_start = find_sublist(passage_no_unk, answer_no_unk, order=answer_order)
                answer_end = answer_start + len(answer_no_unk) - 1 if answer_start >= 0 else -1
                if answer_start < 0:
                    impossible_questions += 1

                if answer_start >= 0 or split != 'train':
                    processed_QA['question'] = question
                    processed_QA['question_no_unk'] = question_no_unk
                    processed_QA['answer'] = raw_answers
                    processed_QA['answer_start'] = answer_start
                    processed_QA['answer_end'] = answer_end
                    processed_QA['id'] = QA['QID']
                    QAs.append(processed_QA)

            # Save processed data
            with open('data/%s/passage/%s|%s' % (split, dataset, PID), 'w') as f:
                assert passage == ' '.join(passage).split(' ')
                f.write(' '.join(passage))
            with open('data/%s/passage_no_unk/%s|%s' % (split, dataset, PID), 'w') as f:
                assert passage_no_unk == ' '.join(passage_no_unk).split(' ')
                f.write(' '.join(passage_no_unk))
            with open('data/%s/datedur_mask/%s|%s' % (split, dataset, PID), 'w') as f:
                f.write(' '.join(datedur_mask))
            with open('data/%s/num_mask/%s|%s' % (split, dataset, PID), 'w') as f:
                f.write(' '.join(num_mask))

            for QA in QAs:
                question = QA['question']
                question_no_unk = QA['question_no_unk']
                answers = QA['answer']
                answer_start = QA['answer_start']
                answer_end = QA['answer_end']
                QID = QA['id']
                with open('data/%s/question/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    assert question  == ' '.join(question).split(' ')
                    f.write(' '.join(question))
                with open('data/%s/question_no_unk/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    assert question_no_unk  == ' '.join(question_no_unk).split(' ')
                    f.write(' '.join(question_no_unk))
                with open('data/%s/answer/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    for answer in answers:
                        f.write('%s\n' % answer)
                with open('data/%s/span/%s|%s|%s' % (split, dataset, PID, QID), 'w') as f:
                    f.write('%d %d' % (answer_start, answer_end))

            print('%s: %d/%d (%.2f%%) \r' % (dataset, i, passage_count, 100*i/passage_count), end='')
        print('\nimpossible_questions: %d' % impossible_questions)
    exit(0)
