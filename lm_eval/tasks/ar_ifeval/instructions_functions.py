import jsonlines
import json
import re
import argparse
import logging
import pandas as pd
from langdetect import detect, LangDetectException
from tqdm import tqdm
import matplotlib.pyplot as plt

arabic_number_words = {
    "واحد": 1, "واحدة": 1, "فقرة": 1, "اثنين": 2, "اثنتين": 2, "فقرتان": 2, "فقرتين": 2, 
    "ثلاثة": 3, "ثلاث": 3, "أربعة": 4, "أربع": 4, "اربع": 4, "خمس": 5, "خمسة": 5, "ستة": 6, "ست": 6, 
    "سبعة": 7, "سبع": 7, "ثمانية": 8, "ثمان": 8, "تسعة": 9, "تسع": 9, "عشرة": 10, "عشر": 10, "جمل": None,
    "الأولى": 1,
    "الثانية": 2,
    "الثالثة": 3,
    "الرابعة": 4,
    "الخامسة": 5,
    "السادسة": 6,
    "السابعة": 7,
    "الثامنة": 8,
    "التاسعة": 9,
    "العاشرة": 10
    }

# Mapping the Arabic text for language to the language code
_LANGUAGE_MAP = {
    'العربية': 'ar',
    'الإنجليزية': 'en',
    'الفرنسية': 'fr',
    'الصينية': 'zh'
}

_COMPARISON_RELATION = ("less than", "at least")

def extract_number_from_text(text, phrases):
    """Extract a number from Arabic words or digits based on a list of phrases."""
    
    # Ensure phrases is a list
    if not isinstance(phrases, list):
        raise ValueError("Phrases should be a list of strings.")
    
    # Create a regular expression pattern for digits and words based on phrases
    phrases_pattern = '|'.join(map(re.escape, phrases))
    digit_pattern = rf"\(?(\d+)\)?\s*({phrases_pattern})"
    word_pattern = rf"({'|'.join(arabic_number_words.keys())})\s*({phrases_pattern})"
    
    # Search for a numeric match
    digit_match = re.search(digit_pattern, text)
    if digit_match:
        return int(digit_match.group(1))
    
    # Search for a word match
    word_match = re.search(word_pattern, text)
    if word_match:
        return arabic_number_words[word_match.group(1)]
    
    # Return None if no match is found
    return None

def check_placeholders(sample, strength):
    """Check if the number of placeholders follows the instruction."""
    response_text = sample["IF_response"]
    prompt_text = sample["instruction_following_prompt"]["prompt"]

    placeholders = re.findall(r"\[.*?\]", response_text)
    num_placeholders = len(placeholders)

    if "موضع " in prompt_text:
        required_placeholders = 1
    elif "موضعين" in prompt_text:
        required_placeholders = 2
    else:
        # Extract the required number of placeholders from the instruction
        required_placeholders = extract_number_from_text(prompt_text, ["مواضع"])

    if required_placeholders is not None:
        # Compare the actual number of placeholders with the required number
        return num_placeholders >= required_placeholders
    else:
        # If no number is found, handle the case appropriately
        print(f"No valid number found for placeholders: {sample['IF_id']}")
        return False

def check_number_bullets(sample, strength):
    """Check if the number of bullet points follows the instruction."""
    response_text = sample["IF_response"]
    prompt_text = sample["instruction_following_prompt"]["prompt"]

    # Regular expression patterns to match bullet points
    bullet_lists_1 = re.findall(r"^\s*\*[^\*].*$", response_text, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-.*$", response_text, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists_1) + len(bullet_lists_2)

    required_bullets = extract_number_from_text(prompt_text, ["نقاط"])

    if required_bullets is not None:
        return num_bullet_lists == required_bullets
    else:
        print(f"No valid number found for bullets: {sample['IF_id']}")
        return False

def check_multiple_sections(sample, strength):
    """Check if the number of sections follows the instruction."""
    response_text = sample["IF_response"]
    prompt_text = sample["instruction_following_prompt"]["prompt"]

    if strength == "loose":
        # Replace multiple symbols with 3 newlines
        response_text = re.sub(r"\[\/\/\]: \# \(Double space\)", "\n\n\n", response_text)
        response_text = re.sub(r"(?:[A-Za-z\u0600-\u06FF]+\s?\d+:)", "\n\n\n", response_text)

    response_text = re.sub(r"<<[A-Za-z\u0600-\u06FF\s]+>>", "\n\n\n", response_text)
    response_text = re.sub(r"(?:(?!\nملاحظة مهمة:)[\n.:](?:[A-Za-z\u0600-\u06FF]+\s?){1,5}[\n.:])", "\n\n\n", response_text)
    response_text = re.sub(r"[#]{2,}", "\n\n\n", response_text)
    response_text = re.sub(r"[/]{2,}", "\n\n\n", response_text)
    response_text = re.sub(r"[-]{2,}", "\n\n\n", response_text)

    # Split based on two or more newlines
    section_splitter_pattern = r"\n{3,}"
    sections = re.split(section_splitter_pattern, response_text.strip())

    # Process each section to replace those with 4 or fewer words with a newline
    processed_sections = []
    for section in sections:
        if strength == "loose" and (section.strip().startswith("ملاحظة") or (section.strip().endswith(":") and len(section.split()) < 13) or len(section.split()) < 10):
            processed_sections.append("\n\n\n")
        else:
            processed_sections.append(section)

    # Join the sections back into the response text
    response_text = "\n\n\n".join(processed_sections)

    # Re-split the processed response text to get the final sections
    sections = re.split(section_splitter_pattern, response_text.strip())
    num_sections = len(sections)

    if "قسمين" in prompt_text or "فقرتين" in prompt_text or "فقرتان" in prompt_text:
        required_sections = 2
    else:
        required_sections = extract_number_from_text(prompt_text, ["أقسام", "فقرات", "نقاط"])

    if required_sections is not None:
        if num_sections == required_sections:
            return True
        else:
            # print("######################################################")
            # print("num_sections:", num_sections)
            # print("multiple sections required_sections:", required_sections)
            # print("IF_id:", sample['IF_id'])
            # print("check_multiple_sections ",sample["IF_response"])
            return False
    else:
        # Uncomment the following line if you want to log cases with no valid number found
        print(f"No valid number found for sections: {sample['IF_id']}")
        return False

def check_highlighted(sample, strength):
    """Check if the highlighted sections follows the instruction."""
    response_text = sample["IF_response"]
    prompt_text = sample["instruction_following_prompt"]["prompt"]

    num_highlights = 0
    highlights = re.findall(r"\*[^\n\*]*\*", response_text)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", response_text)
    for highlight in highlights:
      if highlight.strip("*").strip():
        num_highlights += 1
    for highlight in double_highlights:
        if highlight.startswith("**") and highlight.endswith("**"):
            highlight = highlight[2:-2].strip()
        if highlight:
            num_highlights += 1

    if "قسمين" in prompt_text:
        required_highlights = 2
    elif  re.search(r'قسم(?!\s*مميز)', prompt_text):
        required_highlights = 1
    else:
        required_highlights = extract_number_from_text(prompt_text, ["أقسام"])

    if required_highlights is not None:
        if num_highlights >= required_highlights:
            return True
        else:
            # print("######################################################")
            # print("num_highlights:", num_highlights)
            # print("required_highlights:", required_highlights)
            # print(sample["IF_response"])
            return False
         
    else:
        print(f"No valid number found for highlighted: {sample['IF_id']}")
        return False

def check_title(sample, strength):
    """Checks if the response contains a title in specified formats."""
    response_text = sample["IF_response"]
    prompt_text = sample["instruction_following_prompt"]["prompt"]

    if strength == "strict":
        # Determine the required title format from the prompt
        if "«" in prompt_text and "»" in prompt_text:
            # Titles should be between double angle brackets « »
            pattern = r'«([^«»]+)»'
        elif "\"\"" in prompt_text:
            # Titles should be between double quotation marks " "
            pattern = r'"([^"\\]+)"'
        elif "<" in prompt_text and ">" in prompt_text:
            pattern = r"<([^<>]+)>|>([^><]+)<"
        elif "[[" in prompt_text and "]]" in prompt_text:
            pattern = r"\[([^\[\]]+)\]"
        elif "زاوية مزدوجة" in prompt_text:
            pattern = r'«([^«»]+)»|<([^<>]+)>|>([^><]+)<'
        else:
            # Default pattern: no specific format found
            #print("#######################################################")
            # print(f"No valid title found in prompt: {sample['IF_id']}")
            #print(prompt_text)
            #print(response_text)
            return False
    else: # loose
        # Updated regex pattern to match all formats
        pattern = r'<([^<>]+)>|\'\'([^\']+)\'\'|"""([^"]+)"""|""([^""]+)""|\[([^\[\]]+)\]|\\"([^\"]+)\\"|\\?"([^"\\]+)"|«([^«»]+)»|\.\.\s*([^\.]+?)\s*\.\.|>([^><]+)<'

    re_pattern = re.compile(pattern)

    # Find all matches for all patterns
    matches = re.findall(re_pattern, response_text)

    for match in matches:
        title = next((m for m in match if m), None)
        if title.strip():
            return True

    #print("#######################################################")
    # print(f"No valid title found in prompt: {sample['IF_id']}")
    #print(prompt_text)
    #print(response_text)
    return False

def check_commas(sample, strength):
    """Checks that the response does not contain commas."""
    response_text = sample["IF_response"]

    if not re.search(r'[،,]', response_text): 
        return True
    else:
        #print(f"Comma(s) found in prompt: {sample['IF_id']}")
        return False

def check_json(sample, strength):
    """Check the Json format."""

    response_text = sample["IF_response"]
    # ToDo: do we need actually to check this
    response_text = response_text.replace("\n", "").replace("*", "").replace("{\\\"", "{\"").replace("\": \\\"", "\": \"").replace("\\\"}", "\"}").replace("\\\": \"", "\": \"").replace("\" \"", " ").replace("}}}}", "}}}")
    response_text = re.sub(r'\s{2,}', ' ', response_text)

    # Extract content between first '{' and last '}'
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group()
    else:
        # print(f"No JSON format found in prompt: {sample['IF_id']}")
        #print(response_text)
        return False

    try:
        json.loads(response_text)
    except ValueError:
        response_text = response_text.replace("،", ",")

        pattern = r'(?<=[A-Za-z\u0600-\u06FF])\s?"\s?(?=[A-Za-z\u0600-\u06FF])'
        response_text = re.sub(pattern, r'\\\'', response_text)
        try:
            json.loads(response_text)
        except ValueError:
            #print(f"Invalid JSON format found in prompt: {sample['IF_id']}")
            #print(response_text)
            #print(sample["IF_response"])
            return False

    return True


def number_of_words_checker_at_least(sample, strength):
    """
    Checks if the response in the sample contains the expected number of words based on the prompt.

    Args:
        sample (dict): The sample containing 'instruction_following_prompt' and 'IF_response'.

    Returns:
        bool: True if the response follows the word count instruction, False otherwise.
    """
    prompt = sample.get('instruction_following_prompt', {}).get('prompt')
    response = sample.get('IF_response')

    
    
    response = response.replace("\n\n", " ").strip()
    num_words = len(response.split(" "))

    # Checking for "least" relation
    pattern_least = r'(أجب بما لا يقل عن|أكتب ما لا يقل عن|يجب أن يحتوي (?:ردك|الرد) على ما لا يقل عن)\s*\(?(\d+)\)?\s*كلمة'


    # pattern_least = r'(?:أجب بما لا يقل عن|أكتب ما لا يقل عن|يجب أن تحتوي\s+\S+\s+على\s+(?:ما )?لا يقل عن)\s*\(?(\d+)\)?\s*كلمة'
    match_least = re.search(pattern_least, prompt)
    if match_least:
        min_words = int(match_least.group(2))
        return num_words >= min_words
    else: 
      print("doesn't found pattern for number_word_at_least " +sample['IF_id'] ) 
      return False
 
def number_of_words_checker_at_most(sample, strength):
    """
    Checks if the response in the sample contains at most the specified number of words based on the prompt.

    Args:
        sample (dict): The sample containing 'instruction_following_prompt' and 'IF_response'.

    Returns:
        bool: True if the response follows the "at most" word count instruction, False otherwise.
    """
    prompt = sample.get('instruction_following_prompt', {}).get('prompt')
    response = sample.get('IF_response')

    
    
    response = response.replace("\n\n", " ").strip()
    num_words = len(response.split(" "))

    # Checking for "most" relation (at most)
    pattern_most = r'(أجب بما لا يزيد عن|ولا يزيد عن|أكتب ما لا يزيد عن|يجب أن يحتوي (?:ردك|الرد) على ما لا يزيد عن)\s*\(?(\d+)\)?\s*كلمة'
    match_most = re.search(pattern_most, prompt)
    if match_most:
        max_words = int(match_most.group(2))
        return num_words <= max_words
    else: 
      print("doesn't retun apptern for number_words_at_most  "+ sample['IF_id'])
      # If no "at most" pattern matches, return False
      return False

def count_sentences(s):
    """Counts the number of sentences in a string."""
    sentences = re.split(r'[.?!؟]', s)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return len(sentences)

def number_of_sentences_checker_at_least(sample, strength):
    """Checks if the response has at least N sentences.

    Args:
        sample (dict): A JSON sample containing 'prompt' and 'response' keys.

    Returns:
        bool: True if the response contains at least N sentences, False otherwise.
    """

    pattern = r'(?:يجب أن يحتوي ردك على ما لا يقل عن|أجب بما لا يقل عن) (\d+|ثلاث|اربع|أربع|خمس|ست|سبع|ثمان|تسع|عشر) جمل'

    prompt = sample.get('instruction_following_prompt', {}).get('prompt')
    response = sample.get('IF_response')

    match = re.search(pattern, prompt)
    if match:
        
        match_value = match.group(1)
        if match_value.isdigit():
            num_sentences_threshold = int(match_value)
        else:
            num_sentences_threshold = arabic_number_words.get(match_value)
            if num_sentences_threshold is None:
                #print("doesn't find pattern form sentence_at_least(not a number) "+sample['IF_id'])
                return False
    else:
        print("doesn't find pattern form sentence_at_least "+sample['IF_id'])
        return False

    num_sentences = count_sentences(response)
    return num_sentences >= num_sentences_threshold

def number_of_sentences_checker_at_most(sample, strength):
    """
    Checks if the response in the sample contains at most the specified number of sentences based on the prompt.

    Args:
        sample (dict): The sample containing 'instruction_following_prompt' and 'IF_response'.

    Returns:
        bool: True if the response contains at most the required number of sentences, False otherwise.
    """
    prompt = sample.get('instruction_following_prompt', {}).get('prompt')
    response = sample.get('IF_response')

 

    # Regex pattern for "at most" condition (جمل)
    pattern = r'(?:يجب أن يحتوي ردك على ما لا يزيد عن|أجب بما لا يزيد عن) (\d+|ثلاث|اربع|أربع|خمس|ست|سبع|ثمان|تسع|عشر) جمل'

    # Try to match the pattern in the prompt
    match = re.search(pattern, prompt)
    if match:
        match_value = match.group(1)
        if match_value.isdigit():
            num_sentences_threshold = int(match_value)
        else:
            num_sentences_threshold = arabic_number_words.get(match_value, None)

        if num_sentences_threshold is None:
            print("doesn't find pattern form sentence_at_most(not a number) "+sample['IF_id'])
            return False  # Could not convert Arabic word to number
    else:
        print("doesn't find pattern form sentence_at_most "+sample['IF_id'])
        return False  # Could not extract the number of sentences from the prompt

    # Count the number of sentences in the response
    num_sentences = count_sentences(response)

    # Check if the response contains at most the specified number of sentences
    return num_sentences <= num_sentences_threshold

"""
def paragraph_first_word_check(sample, strength):
    
    # Checks if the response contains the required number of paragraphs and the first word of the nth paragraph.

    # Args:
    #     sample (dict): The sample containing 'instruction_following_prompt' and 'IF_response'.

    # Returns:
    #     bool: True if the response follows the paragraph and first word instruction, False otherwise.
   
    prompt = sample.get('instruction_following_prompt', {}).get('prompt')
    response = sample.get('IF_response')



#     str_check = r'يجب أن تبدأ الفقرة'
#     if str_check not in prompt:
#         print(f"False first_word pattern!! {sample['IF_id']}")
#         return False

#     # Pattern to extract required paragraphs and the nth paragraph with a specific first word
#     pattern = (
#         r'(?:يجب أن يحتوي ردك على|يجب أن يكون هناك) '
#         r'(?:ما لا يقل عن )?(\d+|ثلاث|اربع|أربع|خمس|ست|سبع|ثمان|تسع|عشر) '
#         r'فقرات(?: على الأقل| فقط)?(?:،? الفقرات فقط،)?'
#         r'(?:.*?يتم فصلها بفراغين بين كل فقرة وأخرى\.)?'
#         r'(?:.*?يجب أن تبدأ الفقرة (الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+) بالكلمة "([^"]+)")?'
#     )
#     matches = re.findall(pattern, prompt)

#     num_paragraphs = None
#     nth_paragraph = None
#     first_word = None

#     for match in matches:
#         if match[0]:
#             match_value = match[0]
#             if match_value.isdigit():
#                 num_paragraphs = int(match_value)
#             else:
#                 num_paragraphs = arabic_number_words[match_value]

#         if match[1]:
#             nth_paragraph = arabic_number_words.get(match[1], None)

#         if match[2]:
#             first_word = match[2]

#     if num_paragraphs is None:
#         return False

#     # Split the response into paragraphs
#     paragraphs = re.split(r"\n\n", response.strip())
    
#     # Count non-empty paragraphs
#     valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
#     actual_num_paragraphs = len(valid_paragraphs)

#     # Check if the number of paragraphs is at least the required number
#     if actual_num_paragraphs < num_paragraphs:
#         return False

#     # If a specific nth paragraph and first word is provided, check that
#     if nth_paragraph is not None and first_word is not None:
#         if nth_paragraph <= actual_num_paragraphs:
#             nth_paragraph_content = valid_paragraphs[nth_paragraph - 1]
#             if not nth_paragraph_content.startswith(first_word):
#                 return False
#         else:
#             return False

#     return True
"""

def paragraph_first_word_check(sample,strength):
    """
    Checks if the nth paragraph in the response starts with the specified first word.

    Args:
        sample (dict): The sample containing 'instruction_following_prompt' and 'IF_response'.

    Returns:
        bool: True if the nth paragraph starts with the specified first word, False otherwise.
    """
    prompt = sample.get('instruction_following_prompt', {}).get('prompt')
    response = sample.get('IF_response')

    # Pattern to extract the nth paragraph with a specific first word
    pattern = (
        r'يجب أن تبدأ الفقرة (الأولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+) '
        r'بالكلمة "([^"]+)"'
    )
    match = re.search(pattern, prompt)

    if match:
        # Extract nth_paragraph and first_word from the matched result
        nth_paragraph = match.group(1)
        first_word = match.group(2)
        
        # Convert nth_paragraph from Arabic word to number if necessary
        arabic_number_words = {
            "الأولى": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4, "الخامسة": 5,
            "السادسة": 6, "السابعة": 7, "الثامنة": 8, "التاسعة": 9, "العاشرة": 10
        }
        
        nth_paragraph_index = arabic_number_words.get(nth_paragraph, None)
        # if nth_paragraph_index is None:
            # print("nth_paragraph_index is not a digit or arabic word like الثامنة")
            # print("x",nth_paragraph)
            # # If nth_paragraph is a digit, convert it directly to an integer
            # try:
            #     nth_paragraph_index = int(nth_paragraph)
            # except ValueError:
            #     print("nth_paragraph_index is not a digit or arabic word like الثامنة")
            #     return False

        # Split the response into paragraphs by two newlines
        paragraphs = re.split(r"\n\n", response.strip())
        valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Debugging print statements
        # print("nth_paragraph_index:", nth_paragraph_index)
        # print("first_word:", first_word)
        # print("Paragraphs:", valid_paragraphs)

        # Check if the nth paragraph exists and starts with the specified first word
        if nth_paragraph_index <= len(valid_paragraphs):
            nth_paragraph_content = valid_paragraphs[nth_paragraph_index - 1]
            # print("nth_paragraph_content:", nth_paragraph_content)
            return nth_paragraph_content.startswith(first_word)

def postscript_checker(sample, strength):
    """
    Checks if the response contains the required postscript at the end, as specified in the prompt.

    Args:
        sample (dict): The sample containing 'instruction_following_prompt' and 'IF_response'.

    Returns:
        bool: True if the response contains the expected postscript, False otherwise.
    """
    prompt = sample.get('instruction_following_prompt', {}).get('prompt')
    response = sample.get('IF_response')

  

    # Pattern to match postscript markers in the prompt
    pattern = r'(?:ملاحظة مهمة|ملاحظة هامة|ملاحظة:|ملاحظة|Note:|Note)'
    match = re.search(pattern, prompt)

    if not match:
        print("doesn't find pattern for postscript "+sample['IF_id'])
        return False  # No postscript instruction found in the prompt

    postscript_marker = match.group()

    # Check if the postscript marker is found in the response
    # Assuming the postscript should be at the end of the response
    pattern_postscript_in_response = r'(?<![\w\d]\s)' + re.escape(postscript_marker)
    postscript = re.findall(pattern_postscript_in_response, response, flags=re.MULTILINE)

    # Return True if postscript is found, otherwise False
    return True if postscript else False

def repeat_prompt_then_answer_checker(sample, strength):
    """
    Checks if the response in the sample contains the prompt, then provides an answer.

    Args:
        sample (dict): The sample containing 'instruction_following_prompt' and 'IF_response'.

    Returns:
        bool: True if the response contains the prompt and then provides an answer, False otherwise.
    """
    prompt = sample.get('instruction_following_prompt', {}).get('prompt')
    response = sample.get('IF_response')


    # Extract the part of the prompt up to '?' or '؟' (exclude the question mark)
    match = re.search(r'[؟?]', prompt)
    if match:
        idx = match.start()
        prompt_part = prompt[:idx]  # Exclude the question mark
    else:
        prompt_part = prompt  # No question mark found, use the entire prompt

    if strength == "loose":
        # Remove leading/trailing punctuation, quotation marks, and whitespace
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~؟“”«»"""
        prompt_part = prompt_part.strip(punctuation + ' \t\n\r')
        response_text = response.strip(punctuation + ' \t\n\r')

        # Normalize whitespace and lowercase
        prompt_part = re.sub(r'\s+', ' ', prompt_part).lower()
        response_text = re.sub(r'\s+', ' ', response_text).lower()

        # Remove all punctuation from both strings
        prompt_part_clean = re.sub(rf'[{re.escape(punctuation)}]', '', prompt_part)
        response_text_clean = re.sub(rf'[{re.escape(punctuation)}]', '', response_text)
    else:
        prompt_part_clean = prompt_part
        response_text_clean = response

    # Check if the prompt part appears anywhere in the response
    #id prompt appears anywhere it doesn't follow instruction 
    #double check 
    if  response_text_clean.startswith(prompt_part_clean):
        return True
    else:
        return False

def quotation_checker(sample, strength):
    """
    Checks if the 'IF_response' in the sample is wrapped with double quotation marks.

    Args:
        sample (dict): The sample containing 'IF_response'.

    Returns:
        bool: True if the 'IF_response' is wrapped with double quotation marks, otherwise False.
    """
   
    
    value = sample['IF_response'].strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


def check_include_keywords(sample, strength):
    """Checks if the sample includes the 'include_keywords' category and validates the response."""
    
    def extract_keywords(prompt):
        # Extract keywords between \" using regex
        return re.findall(r'\"(.*?)\"', prompt)

    def check_following(value, keywords):
        """Check if the response contains all the expected keywords."""
        for keyword in keywords:
            if keyword:  # Check if the keyword is not empty
                escaped_keyword = re.escape(keyword)  # Escape special characters
                try:
                    if not re.search(escaped_keyword, value, flags=re.IGNORECASE):
                        #print(f"Keyword not found: '{keyword}' in response")
                        return False
                except re.error as e:
                    # Catch any regex errors and print the problematic keyword
                    print(f"Regex error for keyword when trying to find the keyword from prompt '{keyword}': {e}")
                    return False
        return True

    # Check if 'categories' contains 'include_keywords'
    if 'include_keywords' in sample['instruction_following_prompt']['categories']:
        # Extract the prompt and response
        instruction_following_prompt = sample['instruction_following_prompt']['prompt']
        if_response = sample['IF_response']
        
        # Extract keywords from the prompt
        keywords = extract_keywords(instruction_following_prompt)
        
        # Check if the response follows the instruction and includes all keywords
        return check_following(if_response, keywords)
    else:
        return False  # Return False if 'include_keywords' category is not present

def check_keyword_frequency(sample, strength):
    """Checks the frequency of keywords in a response based on the sample's prompt instructions."""

    def extract_keywords(instruction_following_prompt):
        """Extracts keywords from the instruction following prompt."""
        keyword_pattern = re.findall(r'كلمة "(.*?)"|كلمتي "(.*?)" و"(.*?)"', instruction_following_prompt)
        keywords = []
        for match in keyword_pattern:
            filtered_keywords = list(filter(None, match))
            keywords.extend(filtered_keywords)
        return keywords

    def extract_frequency_and_relation(instruction_following_prompt):
        """Extracts the frequency and relation (at least/less than) from the instruction following prompt."""
        frequency_pattern = re.search(r'على الأقل (\d+)|(\d+) مرات|(\w+) مرات', instruction_following_prompt)
        frequency = 0
        if frequency_pattern:
            if frequency_pattern.group(1):
                frequency = int(frequency_pattern.group(1))
            elif frequency_pattern.group(2):
                frequency = int(frequency_pattern.group(2))
            elif frequency_pattern.group(3):
                frequency = arabic_number_words.get(frequency_pattern.group(3), 0)

        if 'على الأقل' in instruction_following_prompt:
            relation = _COMPARISON_RELATION[1]  # 'at least'
        elif 'على الأكثر' in instruction_following_prompt:
            relation = _COMPARISON_RELATION[0]  # 'less than'
        else:
            relation = _COMPARISON_RELATION[1]  # Default to 'at least'
            # print("Default to 'at least'", sample['IF_id'])
            # print("relation", relation)
        return frequency, relation

    def check_following(value, keywords, frequency, relation):
        """Checks if the response contains the keywords with the required frequency."""
        if len(keywords) == 1:
            actual_occurrences = len(re.findall(keywords[0], value, flags=re.IGNORECASE))
            if relation == _COMPARISON_RELATION[0]:  # "less than"
                return actual_occurrences < frequency
            elif relation == _COMPARISON_RELATION[1]:  # "at least"
                return actual_occurrences >= frequency
        elif len(keywords) == 2:
            actual_occurrences_1 = len(re.findall(keywords[0], value, flags=re.IGNORECASE))
            actual_occurrences_2 = len(re.findall(keywords[1], value, flags=re.IGNORECASE))
            if relation == _COMPARISON_RELATION[0]:  # "less than"
                return actual_occurrences_1 < frequency and actual_occurrences_2 < frequency
            elif relation == _COMPARISON_RELATION[1]:  # "at least"
                return actual_occurrences_1 >= frequency and actual_occurrences_2 >= frequency
        return False

    # Process the sample to check the keyword frequency
    if "keyword_frequency" in sample['instruction_following_prompt']['categories']:
        instruction_following_prompt = sample['instruction_following_prompt']['prompt']
        if_response = sample["IF_response"]

        # Extract keywords, frequency, and relation
        keywords = extract_keywords(instruction_following_prompt)
        frequency, relation = extract_frequency_and_relation(instruction_following_prompt)

        # Check the response against the keywords, frequency, and relation
        result = check_following(if_response, keywords, frequency, relation)

        # Output result
        #print(f"The keyword/s '{keywords}' appear {relation} {frequency} times: {result}")
        return result
    else:
        return False

def check_forbidden_words(sample, strength):
    """Checks if the response does not contain forbidden words specified in the prompt."""
    
    def extract_forbidden_words(instruction_following_prompt):
        """
        Extract forbidden words from the instruction following prompt.
        Looks for patterns like 'لا تقم بتضمين الكلمات المفتاحية' or 'لا تقم بتضمين الكلمات'.
        """
        forbidden_words_pattern = re.findall(r'لا تقم بتضمين الكلمات المفتاحية "(.*?)"|لا تقم بتضمين الكلمات "(.*?)" و"(.*?)"|لا تقم بتضمين الكلمات "(.*?)"', instruction_following_prompt)
        
        forbidden_words = []
        for match in forbidden_words_pattern:
            filtered_words = list(filter(None, match))  # Filter out empty strings
            forbidden_words.extend(filtered_words)
        
        return list(set(forbidden_words))  # Remove duplicates and return unique forbidden words
    
    def check_following(value, forbidden_words):
        """Check if the response does not contain the forbidden words."""
        for word in forbidden_words:
            # Search for whole word matches in a case-insensitive manner
            if re.search(r"\b" + re.escape(word) + r"\b", value, flags=re.IGNORECASE):
                return False  # Forbidden word found
        return True  # No forbidden words found

    # Extract prompt and response from the sample
    instruction_following_prompt = sample['instruction_following_prompt']['prompt']
    if_response = sample["IF_response"]

    # Extract forbidden words from the prompt
    forbidden_words = extract_forbidden_words(instruction_following_prompt)

    # Check if the response contains any forbidden words
    result = check_following(if_response, forbidden_words)

    # Return True if no forbidden words are found, False otherwise
    return result

def check_letter_frequency(sample, strength):
    """Checks if the response meets the letter frequency requirement as per the instruction."""
    def extract_letter_frequency(instruction_following_prompt):
        """Extracts the letter and frequency from the instruction following prompt."""
        letter_pattern = re.search(r'الحرف "(ن|م|ال)"', instruction_following_prompt)
        letter_pattern = re.search(r'الحرف "(ك|ن|م|ال)"', instruction_following_prompt)
        frequency_pattern = re.search(r'(\d+) (مرات|مرة)|(\w+) (مرات|مرة)', instruction_following_prompt)

        letter = letter_pattern.group(1) if letter_pattern else None
        if frequency_pattern:
            if frequency_pattern.group(1):
                frequency = int(frequency_pattern.group(1))
            else:
                frequency = arabic_number_words.get(frequency_pattern.group(2), 1)
        else:
            frequency = 1  # Default to 1 if no frequency is specified
            #print("Default to 'frequency'", sample['IF_id'])

        if "على الأقل" in instruction_following_prompt:
            relation = _COMPARISON_RELATION[1]  # 'at least'
        elif "على الأكثر" in instruction_following_prompt:
            relation = _COMPARISON_RELATION[0]  # 'less than'
        else:
            relation = _COMPARISON_RELATION[1]  # Default to 'at least'
            # print("Default to relation = _COMPARISON_RELATION[1] Default to 'at least'", sample['IF_id'])

        return letter, frequency, relation

    def check_following(value, letter, frequency, relation):
        """Checks that the letter appears in the response with the required frequency."""
        value = value.lower()  # Normalize the response to lowercase
        
        if letter == "ال":
            # Count occurrences of "ال" as a prefix in words
            occurrences = len(re.findall(r"\bال\w+", value))
        else:
            # Count occurrences of standalone letters like "ن" or "م"
            occurrences = value.count(letter)

        if relation == _COMPARISON_RELATION[0]:  # "less than"
            if occurrences < frequency:
                return True
            else:
                #print(sample["IF_id"])
                return False
        else:  # "at least"
            if occurrences >= frequency:
                return True
            else:
                #print(sample["IF_id"])
                return False

    instruction_following_prompt = sample['instruction_following_prompt']['prompt']
    if_response = sample["IF_response"]

    # Extract the letter, frequency, and relation
    letter, frequency, relation = extract_letter_frequency(instruction_following_prompt)

    # Perform the check
    if letter:
        result = check_following(if_response, letter, frequency, relation)
        return result
    
    else:
        print("did not find letter " +sample['IF_id'])
        return False  # Return False if no letter frequency requirement is found

def check_response_language(sample, strength):
    """Checks if the response in the sample is in the expected language."""
    
    def extract_expected_language(instruction_following_prompt):
        """Extracts the expected language from the instruction following prompt."""
        language_pattern = re.search(r'باللغة (العربية|الإنجليزية|الفرنسية|الصينية)', instruction_following_prompt)
        if language_pattern:
            language_name = language_pattern.group(1)
            # print(language_name)
            # print("Default to _LANGUAGE_MAP.get(language_name, 'ar')", sample['IF_id'])
            # print("", sample['instruction_following_prompt']['prompt'])
            # print("language_name", language_name)
            return _LANGUAGE_MAP.get(language_name, 'no_language')  # Default to 'ar' if not found
            
        else: 
            print("no language pattern")
            
            return 'no_pattern'  # Default to Arabic if no specific language is mentioned

    def check_following(value, expected_language):
        """Checks if the detected language matches the expected language."""
        try:
            detected_language = detect(value)
            return detected_language == expected_language
        except LangDetectException as e:
            # logging.error("Unable to detect language for text '%s' due to: %s", value, e) 
            return False

    # Check if the sample has a response language instruction
    if "response_language" in sample['instruction_following_prompt']['categories']:
        instruction_following_prompt = sample['instruction_following_prompt']['prompt']
        if_response = sample["IF_response"]

        # Extract the expected language
        expected_language = extract_expected_language(instruction_following_prompt)
        if expected_language == 'no_pattern':
            print("no_pattern: \n", sample['IF_id'])
        
        if expected_language == 'no_language':
            print("no_pattern: \n", sample['IF_id'])

        # Check if the response follows the expected language
        return check_following(if_response, expected_language)

    return False  # Return False if no "response_language" category

def check_number_paragraphs(sample, strength):
    """Checks if the response contains the expected number of paragraphs."""

    def extract_expected_paragraphs(instruction_following_prompt):
        """Extracts the expected number of paragraphs from the instruction following prompt."""
        paragraphs_pattern = re.search(r'(\d+|[أ-ي]+) فقرات', instruction_following_prompt)
        if paragraphs_pattern:
            number_str = paragraphs_pattern.group(1)
            # print("number_str:\n",number_str)
            if number_str == 'ثلاث' or 'ثلاثة':
                number_str='3'

            if number_str.isdigit():
                return int(number_str)
            else:
                print("Default to _LANGUAGE_MAP.get(language_name, 'ar')", number_str)
                print("Default to _LANGUAGE_MAP.get(language_name, 'ar')", sample['IF_id'])
                return arabic_number_words.get(number_str)  # Default to 1 if not found
        else:
           print("number of paragraph i snot detected from sample "+ sample['IF_id'])
           return 1  # Default to 1 if no specific number is found

    def check_following(value, expected_paragraphs):
        """Checks if the response contains the expected number of paragraphs."""
        dividers = ["***", "* * *"]  # Dividers for splitting paragraphs
        pattern = r'\s?(?:' + '|'.join(re.escape(divider) for divider in dividers) + r')\s?'
        paragraphs = re.split(pattern, value)
        num_paragraphs = len(paragraphs)
        
        # Remove empty paragraphs at the beginning and end
        if not paragraphs[0].strip():
            num_paragraphs -= 1
        if not paragraphs[-1].strip():
            num_paragraphs -= 1

        return num_paragraphs == expected_paragraphs

    # Check if the sample requires checking the number of paragraphs
    if "number_paragraphs" in sample['instruction_following_prompt']['categories']:
        instruction_following_prompt = sample['instruction_following_prompt']['prompt']
        if_response = sample["IF_response"]

        # Extract the expected number of paragraphs
        expected_paragraphs = extract_expected_paragraphs(instruction_following_prompt)

        # Check if the response meets the expected number of paragraphs
        return check_following(if_response, expected_paragraphs)

    return False  # Return False if no "number_paragraphs" category

def check_end_checker(sample, strength):
    """Checks if the sample includes the 'end_checker' category and validates the response."""
    def extract_end_phrase(prompt):
        """Extracts the phrase after specific instruction phrases."""
        # Use regex to extract the text between quotes after the target instruction phrases
        match = re.search(r'(أنهي ردك بهذه العبارة بالضبط|وينتهي بالعبارة|نهاية ردك بهذه العبارة بالضبط)\s*"(.*?)"', prompt)
        if match:
            return match.group(2)  # Return the text inside the quotes
        else:
            print("no match for end_cheker "+sample['IF_id'])
        return None  # Return None if no phrase is found
 
    def check_following(value, end_phrase):
        """Check if the response ends with the expected phrase."""
        value = value.strip().strip("\"").lower()
        end_phrase = end_phrase.strip().lower()
 
        # Check if the response ends with the expected phrase followed by an optional full stop
        return value.endswith(end_phrase) or value.endswith(end_phrase + '.')
 
    # Check if 'categories' contains 'end_checker'
    if 'end_checker' in sample['instruction_following_prompt']['categories']:
        # Extract the prompt and response
        instruction_following_prompt = sample['instruction_following_prompt']['prompt']
        if_response = sample['IF_response']
 
        # Extract the expected end phrase from the prompt
        end_phrase = extract_end_phrase(instruction_following_prompt)
 
        # Check if the response ends with the expected phrase
        if end_phrase:
            return check_following(if_response, end_phrase)
        else:
            print(f"No valid end phrase found in prompt "+sample['If_id'])
            return False
    else:
        return False  # Return False if 'end_checker' category is not present

def check_two_responses(sample, strength):
    """Checks if the sample includes the 'two_responses' category and validates the response."""
 
    def check_following(value):
        """Checks if the response has two different answers.
        Args:
          value: A string representing the response.
        Returns:
          True if two responses are detected and False otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
            len(valid_responses) == 2
            and valid_responses[0].strip() != valid_responses[1].strip()
        )
 
    # Check if 'categories' contains 'two_responses'
    if 'two_responses' in sample['instruction_following_prompt']['categories']:
        # Extract the response
        if_response = sample['IF_response']
 
        # Check if the response follows the instruction for two responses
        return check_following(if_response)
    else:
        return False
