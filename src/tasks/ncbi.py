# define task prompts for various datasets
import re
from .base_task import BaseDataset, BaseTask
import re
import string
import numpy as np
from collections import defaultdict
import random
from datasets import load_dataset

def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)


def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=False):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return:
    correct_chunks: a dict (counter),
                    key = chunk types,
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):

        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)


def get_result(correct_chunks, true_chunks, pred_chunks,
               correct_counts, true_counts, pred_counts, verbose=True):
    """get_result
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type

    print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')

    if nonO_correct_counts > 0:
        print("accuracy: %6.2f%%; (non-O)" % (100 * nonO_correct_counts / nonO_true_counts))
        print("accuracy: %6.2f%%; " % (100 * sum_correct_counts / sum_true_counts), end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f%%" % (prec, rec, f1))
    else:
        print("accuracy: %6.2f%%; (non-O)" % 0)
        print("accuracy: %6.2f%%; " % 0, end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f%%" % (prec, rec, f1))

    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        print("%17s: " % t, end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f%%" %
              (prec, rec, f1), end='')
        print("  %d" % pred_chunks[t])

    return res
    # you can generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    # but I'm not implementing this


def evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_chunks, true_chunks, pred_chunks,
     correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts, verbose)
    return result


def evaluate_conll_file(fileIterator):
    true_seqs, pred_seqs = [], []

    for line in fileIterator:
        cols = line.strip().split()
        # each non-empty line must contain >= 3 columns
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            # extract tags from last 2 columns
            true_seqs.append(cols[-2])
            pred_seqs.append(cols[-1])
    return evaluate(true_seqs, pred_seqs)

class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            yield self._collate_fn(batch_data)

    def _collate_fn(self, batch_data):
        # This function will transform a batch of data into the desired format.
        question, answers = zip(*[(item['question'], item['answer']) for item in batch_data])  # Changed to tags
        return {'question': question, 'answer': answers, }

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    # def split_hf_dataset(self, hf_dataset, train_frac, val_frac):
    #     total_samples = len(hf_dataset)
    #     train_end = int(total_samples * train_frac)
    #     val_end = train_end + int(total_samples * val_frac)
        
    #     train_set = hf_dataset[:train_end]
    #     val_set = hf_dataset[train_end:val_end]
        
    #     return train_set, val_set

    # def set_datasets(self, hf_datasets, train_frac=0.8, val_frac=0.1):
    #     # split the huggingface train set into train and validation
    #     train_set, val_set = self.split_hf_dataset(hf_datasets['train'], train_frac, val_frac)

    #     self.dataset = {
    #         'train': train_set,
    #         'val': val_set,
    #         'test': hf_datasets['test'],
    #         'eval': hf_datasets['eval']
    #     }

class NCBIDataset(BaseDataset):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)
        
class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 
                 task_name = "ncbi",
                 task_description = "Find the disease entity",
                 data_dir='',  
                 seed=42, 
                 
                 post_instruction=True, 
                 TaskDataset=BaseDataset,
                 option_num=5, 
                 **kwargs):
        self.options = {}
        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        TaskDataset=TaskDataset,
                        option_num=option_num,
                        )
        self.answer_format_prompt= "Output the answer in this format:{entity_1,entity_2,....}. If no disease entities are present, please output an empty list in this format: {}."
        
    def load_task_dataset(self, data_dir):
        return load_dataset("ncbi_disease").filter(lambda example: len(example["ner_tags"]) > 0)
    
    @staticmethod
    def extract_entities(tokens, ner_tags):
        entities = []
        entity_tokens = []
        for token, tag in zip(tokens, ner_tags):
            if tag == 1:  # Begin of an entity
                # If there was a previous entity being collected, store it
                if entity_tokens:
                    entities.append(" ".join(entity_tokens))
                    entity_tokens = []
                entity_tokens.append(token)
            elif tag == 2:  # Inside an entity
                entity_tokens.append(token)
            else:  # No entity
                # If there was a previous entity being collected, store it
                if entity_tokens:
                    entities.append(" ".join(entity_tokens))
                    entity_tokens = []

        # Check if there's any leftover entity tokens to be stored
        if entity_tokens:
            entities.append(" ".join(entity_tokens))

        return entities
    
    @staticmethod
    def convert_to_BIO(number_tag):
        # Replace 0 with 'O', 1 with 'B', and 2 with 'I'
        converted_tag = [str(x).replace('0', 'O').replace('1', 'B').replace('2', 'I') for x in number_tag]
        
        # Join the elements with a space
        result = " ".join(converted_tag)
        
        return result
    
    def get_dataloader(self, split, batch_size, shuffle=False):
        if split not in self.dataset:
            raise ValueError(f'Dataset split {split} does not exist.')

        dataset = self.dataset[split]
        return CustomDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def transform_format(self, data_dict):
        transformed_data_dict = {}
        
        for split_name, data_split in data_dict.items():
            print(f"Transforming {split_name} data...")
            
            formatted_data = []
            
            # Check available columns
            available_columns = data_split.column_names
            print(f"Available columns: {available_columns}")
            
            if 'tokens' in available_columns and 'ner_tags' in available_columns:
                for tokens, ner_tags in zip(data_split['tokens'], data_split['ner_tags']):
                    question = tokens
                    answer = self.extract_entities(tokens, ner_tags)
                    formatted_example = {
                        'question': question,
                        'answer': answer
                    }
                    formatted_data.append(formatted_example)
                transformed_data_dict[split_name] = formatted_data
            else:
                print(f"Columns 'tokens' and/or 'ner_tags' not found in {split_name} data_split.")
        
        return transformed_data_dict

    def get_random_dataloader(self, size, batch_size, shuffle=False):
        if self.TaskDataset is None:
            self.TaskDataset = BaseDataset
        
        random.shuffle(self.all_train_set)
        dataset = self.build_task_dataset(self.all_train_set[:size], TaskDataset=self.TaskDataset)
            
        return CustomDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def build_forward_prompts_completion(self, questions, cur_propmt):
        return super().build_forward_prompts_completion(questions, cur_propmt)
    
    def clean_labels(self, answers):
        cleaned_answers = []

        for answer in answers:
            # Check if the current answer is a string
            if isinstance(answer, str):
                cleaned_answers.append(answer.lower())
            # Check if the current answer is a list
            elif isinstance(answer, list):
                # Convert each string inside the list to lowercase
                cleaned_list = [item.lower() for item in answer if isinstance(item, str)]
                cleaned_answers.append(cleaned_list)
            else:
                cleaned_answers.append(answer)  # Handle other data types if needed

        return tuple(cleaned_answers)

    @staticmethod
    def post_processing(sentence, predicted_ents):
        post_predicted_ents = []
        for ent in predicted_ents:
            ent = ent.replace(',', '')
            if ' ' + ent + ' ' in ' ' + sentence.lower() + ' ':
                post_predicted_ents.append(ent)
        return list(set(post_predicted_ents))

    # Your revised create_bio_preds

    def create_bio_preds(self, questions, predictions):
        bio_preds = []
        for q, p in zip(questions, predictions):
            sent = ' '.join(q).lower()
            predicted_entities = [ent.strip().lower() for ent in p]
            
            post_predicted_ents = self.post_processing(sent, predicted_entities)
            post_predicted_ents.sort(key=len, reverse=True)
            
            bio_pred_seq = ' ' + sent + ' '
            for pred_ent in post_predicted_ents:
                pred_ent = re.sub('\s+', ' ', pred_ent)
                
                if pred_ent != '':
                    pred_bios = ['I' for _ in pred_ent.split()]
                    pred_bios[0] = 'B'
                    pred_bios = ' '.join(pred_bios)
                    bio_pred_seq = bio_pred_seq.replace(' ' + pred_ent + ' ', ' ' + pred_bios + ' ')
            
            bio_pred_seq = ' '.join(['O' if (w != 'B' and w != 'I') else w for w in bio_pred_seq.split()])
            bio_pred_seq = bio_pred_seq.strip()
            bio_preds.append(bio_pred_seq)
        return bio_preds
    
    @staticmethod
    def conlleval_eval(true, preds):
        true = [[t[0] + '-X' for t in s.split()] for s in true]
        preds = [[t[0] + '-X' for t in s.split()] for s in preds]
        if not true or not preds:
            return 0, 0, 0
        true = np.concatenate(true)
        preds = np.concatenate(preds)
        print(true)
        print(preds)
        prec, recall, f1 = evaluate(true, preds)

        return f1, prec, recall
    
    def cal_correct(self, preds, labels):
        # Assuming both preds and labels are lists of lists
        if len(preds) != len(labels):
            print("Mismatched length between preds and labels")
            return []
            
        comparison_results = []
        for pred, label in zip(preds, labels):
            # Compare each pair of lists
            comparison_results.append(int(pred == label))
        
        return comparison_results

    def cal_metric(self, preds, labels, questions):
        '''
            <task specific>
            return 1 number / tuple of metrics
        '''
        batch_bio_preds = self.create_bio_preds(questions, preds)
        batch_bio_label = self.create_bio_preds(questions, labels)

        new_list = [s.upper() for s in batch_bio_label]
        f1, precision, recall = self.conlleval_eval(new_list, batch_bio_preds)
        return f1, precision, recall

    def clean_response(response):
        # regex pattern to capture disease entity phrases enclosed in curly braces
        entity_pattern = re.compile(r'\{(.*?)\}', re.IGNORECASE)
        
        entities = []
        matches = entity_pattern.findall(response)

        for match in matches:
            entities.extend([e.strip().lower() for e in match.split(',') if e.strip()])
                    
        return entities
    
    
    @staticmethod
    def clean_response(response):
        # regex pattern to capture disease entity phrases enclosed in curly braces
        entity_pattern = re.compile(r'\{(.*?)\}', re.IGNORECASE)
        
        entities = []
        matches = entity_pattern.findall(response)

        for match in matches:
            # Remove or standardize spaces around slashes and hyphens
            normalized_match = re.sub(r'\s*([/-])\s*', r' \1 ', match)
            
            # Normalize to lowercase and strip whitespace
            entities.extend([e.strip().lower() for e in normalized_match.split(',') if e.strip()])
                    
        return entities

    
    def batch_clean_responses(self, responses):
        if not isinstance(responses, list):
            responses = list(responses)
        batch_answers = []
        for response in responses:
            batch_answers.append(self.clean_response(response))
        return batch_answers