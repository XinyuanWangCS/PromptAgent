self_eval_prompt_template_v0 = """
I'm trying to write a prompt for a language model on a dataset. Here are some question examples and their labels in the dataset:
{examples_str}

My prompt is:
{cur_prompt}

Based on the above information, evaluate the prompt with an integer score between 0 and 10, and adhering to these guidelines:
1. Think step by step to see if the prompt can help the model answer these questions and get the correct answers.
2. The larger the score is, the more helpful the prompt is:
    0 means this prompt is unable to help answer any questions;
    5 means this prompt is able to help answer questions in the questions above;
    10 means this prompt can be generalized to help answer any similar questions.
3. Wrap the score with <START> and <END>
""".strip()

self_eval_prompt_template_v1 = """
I'm trying to write a prompt for a language model on a dataset. Here are some question examples and their labels in the dataset:
{examples_str}

My prompt is:
{cur_prompt}

Based on the above information, evaluate the prompt with an integer score between 0 and 10, adhering to these guidelines:
1. Consider whether the prompt can assist the model in accurately answering the questions step by step.
2. Assign a score based on the prompt's utility:
    - 0 means the prompt doesn't aid in answering any questions.
    - 5 means the prompt assists in answering the given questions.
    - 10 suggests that the prompt is generalizable to similar questions.
3. Wrap the score with <START> and <END>.
""".strip()

example_template_v0 = """
<{index}>
Question:
{question}
Label: {label}
"""

self_eval_prompt_template_v2 = """
I'm trying to write a prompt for a language model on a dataset. Here are some question examples and their labels in the dataset:
{examples_str}

My prompt is:
{cur_prompt}

To evaluate the given prompt:
1. Provide a brief explanation or reasoning for your score.
2. Analyze the specific details and nuances of the example questions.
3. Consider how well the given prompt assists the model in addressing the nuances and details from the example questions.
4. Assign a score based on the prompt's utility:
    - 0 means the prompt doesn't address or aid in answering any of the nuances in the questions.
    - 5 means the prompt provides guidance that assists in answering the specific questions provided.
    - 10 means the prompt is versatile and can be generalized to assist in answering not just the given questions but also other similar questions.


Now, based on your analysis, wrap the score with <START> and <END>, and provide the accompanying explanation.
""".strip()

self_eval_prompt_template_v3 = """
I'm trying to write a prompt for a language model on a dataset. You should evaluate the prompt. The prompt is:
{cur_prompt}

This prompt helps the model produce these correct answers:
{correct_str}

This prompt leads to these incorrect answers:
{error_str}

To evaluate the given prompt:
1. Analyze the specific details and nuances of the example questions.
2. Consider how well the given prompt assists the model in addressing the nuances and details from the example questions.
3. Provide an explanation or reasoning before your score.
4. Assign a score based on the prompt's utility (higher score means better prompt):
    - 0 means the prompt doesn't address or aid in answering any of the nuances in the questions.
    - 5 means the prompt provides guidance that assists in answering the specific questions provided.
    - 10 means the prompt is versatile and can be generalized to assist in answering not just the given questions but also other similar questions.

Finally, wrap the score with <START> and <END>.
""".strip()

self_eval_prompt_template_v4 = """
I'm trying to write a prompt for a language model on a dataset. Here are some examples of questions and labels in the dataset:
{examples_str}

You are going to compare two prompts:
Propmt 1: {parent_prompt}
Prompt 2: {cur_prompt}

You should compare them following these guidelines:
1. Consider how well the prompt assists the model in addressing the nuances and details from the example questions.
2. Consider how well the prompt generalizes to other similar questions.
3. Provide an step by step explanation or reasoning before giving your result.
4. Assign a score based on all the information:
    - 0 means prompt 1 much better than promp 2
    - 1 means prompt 1 relatively better than promp 2
    - 2 means prompt 1 is almost the same as promp 2
    - 3 means prompt 2 relatively better than promp 1
    - 4 means prompt 2 much better than promp 1
5. Finally, wrap the score with <START> and <END>.
""".strip()

self_eval_prompt_template_v5 = """
I'm trying to write a prompt for a language model on a dataset. You should evaluate the prompt. Here are some question examples and their labels in the dataset:
{examples_str}

The prompt is:
{cur_prompt}

This prompt helps the model produce these correct answers:
{correct_str}

This prompt leads to these incorrect answers:
{error_str}

To evaluate the given prompt and follow these guidelines:
1. Analyze the specific details and nuances of the example questions.
2. Consider how well the given prompt assists the model in addressing the nuances and details from the example questions.
3. Provide an explanation or reasoning before your score.
4. Assign a score based on the prompt's utility (higher score means better prompt):
    - 0 means the prompt doesn't address or aid in answering any of the nuances in the questions.
    - 5 means the prompt provides guidance that assists in answering the specific questions provided.
    - 10 means the prompt is versatile and can be generalized to assist in answering not just the given questions but also other similar questions.
5. Finally, wrap the score with <START> and <END>.
""".strip()

self_eval_prompt_template_v6 = """
You are tasked with comparing two prompts for a language model based on a dataset. Here are some example questions and labels from the dataset:
{examples_str}

You must evaluate:
Propmt 1: {parent_prompt}
Prompt 2: {cur_prompt}

Follow these guidelines during your comparison:
1. Assess how effectively each prompt helps the model address the nuances and specifics of the example questions.
2. Determine how well each prompt can generalize to other questions of a similar nature.
3. Provide a step-by-step explanation for your reasoning.
4. Based on the above, give a score:
    - 1: Prompt 1 is significantly better than Prompt 2
    - 2: Prompt 1 is somewhat better than Prompt 2
    - 3: Both prompts are roughly equivalent in effectiveness
    - 4: Prompt 2 is somewhat better than Prompt 1
    - 5: Prompt 2 is significantly better than Prompt 1
5. Finally, wrap the score with <START> and <END>.
""".strip()

