import json
import random
import sys

import torch
from pyswip import Prolog
from problog.program import PrologString
from problog.core import ProbLog
from problog import get_evaluatable
from tqdm import tqdm

from neurasp import NeurASP
from neurasp_clevr.network import Net
from utils.question_encoder import encode_question
from utils.problog import termPath2dataList, func_to_asp

with open("data/CLEVR_v1.0/questions/CLEVR_val_sample_15000.json") as fp:
    questions = json.load(fp)
    # print(questions)

domain = ['LaGraMeCy', 'LaGraMeSp', 'LaGraMeCu', 'LaGraRuCy', 'LaGraRuSp', 'LaGraRuCu', 'LaBlMeCy', 'LaBlMeSp', 'LaBlMeCu', 'LaBlRuCy',
          'LaBlRuSp', 'LaBlRuCu', 'LaBrMeCy', 'LaBrMeSp', 'LaBrMeCu', 'LaBrRuCy', 'LaBrRuSp', 'LaBrRuCu', 'LaYeMeCy', 'LaYeMeSp',
          'LaYeMeCu', 'LaYeRuCy', 'LaYeRuSp', 'LaYeRuCu', 'LaReMeCy', 'LaReMeSp', 'LaReMeCu', 'LaReRuCy', 'LaReRuSp', 'LaReRuCu',
          'LaGreMeCy', 'LaGreMeSp', 'LaGreMeCu', 'LaGreRuCy', 'LaGreRuSp', 'LaGreRuCu', 'LaPuMeCy', 'LaPuMeSp', 'LaPuMeCu', 'LaPuRuCy',
          'LaPuRuSp', 'LaPuRuCu', 'LaCyMeCy', 'LaCyMeSp', 'LaCyMeCu', 'LaCyRuCy', 'LaCyRuSp', 'LaCyRuCu', 'SmGraMeCy', 'SmGraMeSp',
          'SmGraMeCu', 'SmGraRuCy', 'SmGraRuSp', 'SmGraRuCu', 'SmBlMeCy', 'SmBlMeSp', 'SmBlMeCu', 'SmBlRuCy', 'SmBlRuSp', 'SmBlRuCu',
          'SmBrMeCy', 'SmBrMeSp', 'SmBrMeCu', 'SmBrRuCy', 'SmBrRuSp', 'SmBrRuCu', 'SmYeMeCy', 'SmYeMeSp', 'SmYeMeCu', 'SmYeRuCy',
          'SmYeRuSp', 'SmYeRuCu', 'SmReMeCy', 'SmReMeSp', 'SmReMeCu', 'SmReRuCy', 'SmReRuSp', 'SmReRuCu', 'SmGreMeCy', 'SmGreMeSp',
          'SmGreMeCu', 'SmGreRuCy', 'SmGreRuSp', 'SmGreRuCu', 'SmPuMeCy', 'SmPuMeSp', 'SmPuMeCu', 'SmPuRuCy', 'SmPuRuSp', 'SmPuRuCu',
          'SmCyMeCy', 'SmCyMeSp', 'SmCyMeCu', 'SmCyRuCy', 'SmCyRuSp', 'SmCyRuCu']


with open("utils/theory_problog.lp", "r") as fp:
    theory = fp.read()

epoch = sys.argv[1]
conf = sys.argv[2]
k = sys.argv[3]
directory = sys.argv[4]

correct = 0
incorrect = 0
invalid = 0
total = 0

factsDict = termPath2dataList('img data/CLEVR_v1.0/images/val', 480, domain, epoch, conf, k)
# print(factsDict)


print(f"\nEpoch: {epoch}, Confidence: {conf}, Disjunction Size: {k}, Directory: {directory}")


for q in tqdm(questions):

    img_file = q['image_filename']
    # print('The name',img_file)
    incumbent_facts = factsDict[img_file]
    # print(factsDict)

    # print(f"Facts for {img_file}:")
    # print(incumbent_facts)
    # print("\n---\n")
    
    # print(q['program'])
    # program = theory
    # program += '\n'
    # program += func_to_asp(q["program"]) 
    # program += '\n'
    # program += incumbent_facts
    # program += '\n'
    # program += 'query(ans(X)).'
    # print(program)
    asp_program = theory + '\n' + func_to_asp(q["program"]) + '\n' + incumbent_facts + '\nquery(ans(X)).'

    # print(q['answer'])  
    correct_answer = q['answer']
    if correct_answer == 'no':
        correct_answer = 'false'
    if correct_answer == 'yes':
        correct_answer = 'true'
    
    problog_result = get_evaluatable().create_from(asp_program).evaluate()
    if problog_result:
        sorted_result = dict(sorted(problog_result.items(), key=lambda x: x[1], reverse=True))
        top_answer = str(list(sorted_result.keys())[0]).split('(')[1].split(')')[0]
        print(f"Predicted Answer: {top_answer}")
    else:
        print(f"No answer returned by ProbLog for {img_file}.")
        problog_answer = None
    
    if str(correct_answer).lower() == str(top_answer).lower():
        correct+=1
    elif top_answer == None or top_answer == 'X2':
        invalid+=1
    else:
        incorrect+=1
    total+=1


print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")
