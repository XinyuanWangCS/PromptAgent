# These prompts are modified based on Automatic Prompt Optimization with "Gradient Descent" and Beam Search
# https://arxiv.org/abs/2305.03495

gradient_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{example_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
""".strip()

optimize_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{example_string}

Based on these errors, the problems with this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write {steps_per_gradient} new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.

The new prompts are:
""".strip()

example_template = """
<{index}> 
The model's input is:
{question}

The model's response is: 
{response}

The correct label is: {label}
The model's prediction is: {prediction}.
"""

optimize_prompt_tempelate_single = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{example_string}

Based on these errors, the problems with this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write 1 new prompt following these guidelines:
1. The new prompt should solve the current prompt's problems.
2. The new prompt should consider the list of prompts and evolve based on the current prompt.
3. The new prompt should be wrapped with <START> and <END>.

The new prompts is:
""".strip()

ascend_gradient_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{example_string}

For each example, carefully examine each question and correct answer step by step, provide comprehensive and different reasons why the prompt leads to the correct answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
""".strip()

ascend_optimize_prompt_tempelate = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{example_string}

Based on these examples, the advantages of this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write {steps_per_gradient} new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.

The new prompts are:
""".strip()

ascend_optimize_prompt_tempelate_single = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{example_string}

Based on these examples, the advantages of this prompt and the reasons are:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write 1 new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.

The new prompts is:
""".strip()