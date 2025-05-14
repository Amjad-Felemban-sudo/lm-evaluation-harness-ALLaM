import dataclasses
from typing import Dict, Optional, Union

from lm_eval.tasks.ar_ifeval.instructions_functions import *
from lm_eval.utils import eval_logger

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

categories_checks = [
        ("number_placeholder", check_placeholders),
        ("number_bullets", check_number_bullets),
        ("multiple_sections", check_multiple_sections),
        ("minimum_number_highlighted_section", check_highlighted),
        ("title", check_title),
        ("no_commas", check_commas),
        ("json_format", check_json),
        ("number_words_at_least", number_of_words_checker_at_least),
        ("number_words_at_most", number_of_words_checker_at_most),
        ("number_sentences_at_least", number_of_sentences_checker_at_least),
        ("number_sentences_at_most", number_of_sentences_checker_at_most),
        ("first_word_in_i-th_paragraph", paragraph_first_word_check),
        ("postscript", postscript_checker),
        ("repeat_prompt", repeat_prompt_then_answer_checker),
        ("quotation", quotation_checker),
        ("include_keywords", check_include_keywords),
        ("keyword_frequency", check_keyword_frequency),
        ("forbidden_words", check_forbidden_words),
        ("letter_frequency", check_letter_frequency),
        ("response_language", check_response_language),
        ("number_paragraphs", check_number_paragraphs),
        ("check_end", check_end_checker),
        ("two_responses", check_two_responses)
    ]

valid_categories = {category for category, _ in categories_checks}

def process_sample(doc, response, strength):
    # print('===================', type(doc["instruction_following_prompt"]))
    # print(doc["instruction_following_prompt"])
    # print(doc.keys())
    sample = {"instruction_following_prompt": {'prompt': doc['prompt'], 'categories': doc['instruction_list']}}
    for i, key in enumerate(doc):
        if i > 2:
            break
        sample[key] = doc[key]
    sample['IF_response'] = response
    print(sample)
    sample_categories = sample["instruction_following_prompt"]["categories"]

    # Filter relevant categories
    matching_categories = [category for category in sample_categories if category in valid_categories]

    is_following_list = []
    # Generate all_responses for 'loose' strength
    if strength == 'loose':
        if_response = sample['IF_response']
        r = if_response.split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()

        revised_response = if_response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        
        all_responses = [
            if_response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]
    else:
        all_responses = [sample['IF_response']]

    for category, check_function in categories_checks:
        if category in matching_categories:
            passed = False
            if strength == 'loose':
                for response in all_responses:
                    # Temporarily replace sample['IF_response'] with response
                    original_response = sample['IF_response']
                    sample['IF_response'] = response

                    result = check_function(sample, strength)
                    if result:
                        passed = True
                        sample['IF_response'] = original_response  # Restore original response
                        break
                    sample['IF_response'] = original_response  # Restore original response
            else:
                # For 'strict' strength
                passed = check_function(sample, strength)

            if passed:
                is_following_list.append(True)
            else:
                is_following_list.append(False)
    return OutputExample(
    instruction_id_list=doc['instruction_list'],
    prompt=doc['prompt'],
    response=response,
    follow_all_instructions=all(is_following_list),
    follow_instruction_list=is_following_list,
    )
        # If the sample passes all relevant checks, add it to the list
        # if relevant_checks_pass:
        #     passing_samples.append(sample)

        # # Calculate and print the accuracy for each category
        # weighted_numerator = 0
        # weighted_denominator = 0
        # for category, counts in category_accuracy.items():
        #     total = counts['correct'] + counts['wrong']
        #     weighted_numerator += counts['correct']
        #     weighted_denominator += total

        # weighted_average_accuracy = weighted_numerator / weighted_denominator * 100
        # print(f"Accuracy of the inst. following {strength}: {weighted_average_accuracy:.2f}%")

def process_results(doc, results):

    response = results[0]
    out_strict = process_sample(doc, response, 'strict')
    out_loose = process_sample(doc, response, 'loose')


    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc
