import argparse
import json
import random

import clingo
from clingo.symbol import SymbolType
from tqdm import tqdm

from utils.question_encoder import encode_question
from utils.utils import help_messages, AnswerMode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--facts', type=str, required=True, help=help_messages['facts'])

    parser.add_argument('-q', '--questions', type=str,
                        default='data/CLEVR_v1.0/questions/CLEVR_val_questions.json', help=help_messages['questions'])

    parser.add_argument('-o', '--out', type=str, required=True, help=help_messages['results_out'])

    parser.add_argument('-t', '--theory', type=str, required=True,
                        help='File (.lp) containing additional ASP rules, e.g. for spatial reasoning')

    parser.add_argument('--answer_mode', type=AnswerMode, default=AnswerMode.single,
                        choices=list(AnswerMode), help=help_messages['answer_mode'])

    args = parser.parse_args()

    # Load facts from specified file
    with open(args.facts) as fp:
        facts = json.load(fp)['facts']
        #print(len(facts))

    # Load questions from specified file
    with open(args.questions) as questions_file:
        questions = json.load(questions_file)["questions"]
        #print(questions['image_index'])

    with open(args.theory, "r") as theory_file:
        theory = theory_file.read()

    total = 0
    correct = 0
    incorrect = 0
    invalid = 0

    for q in tqdm(questions):
        # print('Image: ', q)
        q_encoding = encode_question(q["program"])

        program = '\n'.join(facts[str(q['image_index'])]) + "\n" + q_encoding + "\n" + theory

        answer_candidates = set()
        model_candidates = []
        answer_type = ""

        # Set up clingo and ground/solve
        ctl = clingo.Control(['--warn=none', '-t', '8'])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        ctl.solve(on_model=lambda m: model_candidates.append(m.symbols(shown=True)))

        if len(model_candidates) > 0:
            val = model_candidates[-1][0].arguments[0]
            if val.type == SymbolType.Number:
                val = str(val.number)
            elif val.type == SymbolType.Function:
                if val.name in ['true', 'false']:
                    val = 'no' if val.name == 'false' else 'yes'
                else:
                    val = val.name
            answer_candidates.add(val)

        # Check if computed answer(s) and ground truth are the same
        if answer_candidates:
            # Break ties by picking a random answer
            # Occurs if there are multiple optimal answer sets, which can happen due to reduced precision introduced by rounding)
            guess = [random.choice(list(answer_candidates))]

            assert len(guess) == 1

            ground_truth = str(q["answer"])

            if ground_truth in guess:
                correct += 1
            else:
                incorrect += 1
        else:
            invalid += 1

        total += 1

    print(f"Correct: {correct}/{total} ({correct / total * 100:.2f})")
    print(f"Incorrect: {incorrect}/{total} ({incorrect / total * 100:.2f})")
    print(f"Invalid: {invalid}/{total} ({invalid / total * 100:.2f})")
