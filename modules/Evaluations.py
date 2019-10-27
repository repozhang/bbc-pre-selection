# from pythonrouge.pythonrouge import Pythonrouge
import nltk.translate.bleu_score as bleu
# from pyrouge import Rouge155
from tempfile import *
import os
import codecs
import logging
# from pyrouge.utils import log

# log.get_global_console_logger().setLevel(logging.WARNING)

# def rouge(hypothesises, references):
#     dir=mktemp()
#     os.mkdir(dir)
#     sys_dir=os.path.join(dir, 'sys')
#     os.mkdir(sys_dir)
#     ref_dir = os.path.join(dir, 'ref')
#     os.mkdir(ref_dir)
#
#     uppercase=['A','B','C','D','E','F','G','H','I','J']
#
#     for i in range(len(hypothesises)):
#         sys_file = codecs.open(os.path.join(sys_dir, 'sys.'+str(i)+'.txt'), "w", "utf-8")
#         sys_file.write(hypothesises[i].replace('<','___').replace('>','___'))
#         sys_file.close()
#
#         for j in range(len(references[i])):
#             ref_file = codecs.open(os.path.join(ref_dir, 'ref.'+uppercase[j]+'.' + str(i) + '.txt'), "w", "utf-8")
#             ref_file.write(references[i][j].replace('<','___').replace('>','___'))
#             ref_file.close()
#
#     r = Rouge155()
#     r.system_dir = sys_dir
#     r.model_dir = ref_dir
#     r.system_filename_pattern = 'sys.(\d+).txt'
#     r.model_filename_pattern = 'ref.[A-Z].#ID#.txt'
#
#     scores = r.convert_and_evaluate()
#     # print(scores)
#     scores = r.output_to_dict(scores)
#     return scores

# def rouge(hypothesises, references):
#     b_systems = []
#     b_references = []
#     for i in range(len(hypothesises)):
#         hypothesis = hypothesises[i]
#         reference = [[r] for r in references[i]]
#         b_systems.append([hypothesis])
#         b_references.append(reference)
#
#     rouge = Pythonrouge(summary_file_exist=False,
#                         summary=b_systems, reference=b_references,
#                         n_gram=2, ROUGE_SU4=True, ROUGE_L=True, ROUGE_W_Weight=1.2,
#                         recall_only=False, stemming=True, stopwords=False,
#                         word_level=True, length_limit=False, length=75,
#                         use_cf=True, cf=95, scoring_formula='average',
#                         resampling=True, samples=1000, favor=True, p=0.5)
#     scores = rouge.calc_score()
#
#     print(scores)
#     return scores

def distinct(self, hypothesises):
    scores = dict()
    unigram = set()
    unigram_count = 0
    bigram = set()
    bigram_count = 0
    for hypothesis in hypothesises:
        words = hypothesis.split(' ')
        unigram_count += len(words)
        for i in range(len(words)):
            unigram.add(words[i])
            if i < len(words) - 1:
                bigram.add(words[i] + ' ' + words[i + 1])
                bigram_count += 1
    scores['unigram'] = len(unigram)
    scores['unigram_count'] = unigram_count
    scores['distinct-1'] = len(unigram) / unigram_count
    scores['bigram'] = len(bigram)
    scores['bigram_count'] = bigram_count
    scores['distinct-2'] = len(bigram) / bigram_count
    # print(scores)
    return scores

def perplexity(hypothesises, probabilities):
    scores=dict()
    avg_perplexity=0
    for i in range(len(hypothesises)):
        perplexity = 1
        N = 0
        for j in len(hypothesises[i]):
            N += 1
            perplexity = perplexity * (1/probabilities[i,j].item())
        perplexity = pow(perplexity, 1/float(N))
        avg_perplexity+=perplexity
    scores['perplexity'] =avg_perplexity/len(hypothesises)
    # print(scores)
    return scores

def sentence_bleu(hypothesises, references):
    scores = dict()
    avg_bleu=0
    for i in range(len(hypothesises)):
        avg_bleu+=bleu.sentence_bleu([r.split(' ') for r in references[i]], hypothesises[i].split(' '))
    scores['sentence_bleu'] = avg_bleu/len(hypothesises)
    # print(scores)
    return scores

def corpus_bleu(hypothesises, references):
    scores = dict()
    b_systems = []
    b_references = []
    for i in range(len(hypothesises)):
        b_systems.append(hypothesises[i].split(' '))
        b_references.append([r.split(' ') for r in references[i]])
    scores['corpus_bleu'] = bleu.corpus_bleu(b_references, b_systems)
    # print(scores)
    return scores

