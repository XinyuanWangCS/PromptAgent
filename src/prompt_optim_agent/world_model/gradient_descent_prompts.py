'''
original prompts in paper
'''
gradient_prompt_tempelate_base0 = """
I'm trying to write a zero-shot prompt to solve math problems.

My current prompt is:
"{prompt}"

But this prompt gets the following examples wrong:
{error_string}

give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
""".strip()

optimize_prompt_tempelate_base0 = """
I'm trying to write a zero-shot math problem solver.

My current prompt is:\n{cur_prompt}

But it gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>.

The {steps_per_gradient} new prompts are:
""".strip()

error_example_template_base0 = '<{index}> Question: {question} \nLabel: {label} Prediction: {prediction}\n'

'''
prompt base1:
    1. error_example: add response and gt_answer
    2. number of new prompts

problem:
    1. prompt involves specific question information
'''
error_example_template_base1 = """
<{index}> Question: {question}
The ground truth answer is: {gt_answer} The answer is {label}.
The predicted answer is: {response} The answer is {prediction}.\n
"""

optimize_prompt_tempelate_base1 = """
I'm trying to write a zero-shot math problem solver.

My current prompt is:\n{cur_prompt}

But it gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>.

The {steps_per_gradient} new prompts is:
""".strip()

'''
prompt base2:
    1. error_example: avoid involving specific question information.
    2. change to "are", change number to 2 but only consider the first one
problem:
    1. flags=re.DOTALL includes \n
    2. still involve the question's information
    3. test cases should be the same, let's make a 20 size test_set
'''
optimize_prompt_tempelate_base2 = """
I'm trying to write a zero-shot math problem solver.

My current prompt is:\n{cur_prompt}

But it gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>.

Pay attention that the new prompt should be general to all similar questions and do not involve the information in the above error cases.

The {steps_per_gradient} new prompts are:
""".strip()

'''
prompt alpha0
'''
gradient_prompt_tempelate_alpha0 = """
I'm trying to write a zero-shot prompt for a language model. The input of this model is in the format of "<prompt> <question>". 

My current prompt is:
"{cur_prompt}"

But this prompt gets the following examples wrong:
{error_string}

Compare the correct and predicted answers above. Give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
""".strip()

optimize_prompt_tempelate_alpha0 = """
I'm trying to write a zero-shot prompt for a language model. The input of this model is in the format of "<question> <prompt>". 

My current prompt is:\n{cur_prompt}

But it gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>.

Try to write the prompts that can be applied to all the similar questions and do not involve the information in the above error cases.

The {steps_per_gradient} new prompts are:
""".strip()

error_example_template_alpha0 = """
<{index}> Question: {question}
The correct answer is: 
{gt_answer}
The correct result is: {label}.

The predicted answer is: 
{response}
The predicted result is {prediction}.
"""

'''
prompt alpha1
'''
gradient_prompt_tempelate_alpha1 = """
I'm trying to write a zero-shot prompt for a language model. The input of this model is in the format of:
"Prompt: <prompt>\nQuestion: <question>\nProvide a response based on the preceding prompt and question, and put your answer within tag <Ans> and </Ans>.".

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

Compare the correct and predicted answers above. Explain the reason why the prompt could have gotten these examples wrong.
""".strip()

optimize_prompt_tempelate_alpha1 = """
I'm trying to write a zero-shot prompt for a language model. The input of this model is in the format of:
"Prompt: <prompt>\nQuestion: <question>\nProvide a response based on the preceding prompt and question, and put your answer within tag <Ans> and </Ans>.".

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors but do not involve the information in the above error cases.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

error_example_template_alpha1 = """
<{index}> Question: {question}
The correct answer is: 
{gt_answer}
The correct result is: {label}.

The predicted answer is: 
{response}
The predicted result is {prediction}.
"""

'''
prompt alpha2
'''
gradient_prompt_tempelate_alpha2 = """
I'm trying to write a zero-shot prompt for a language model. The input of this model is in the format of:
"<prompt> <question> Answer:"

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

Compare the correct and predicted answers above. Explain the reason why the prompt could have gotten these examples wrong.
""".strip()

optimize_prompt_tempelate_alpha2 = """
I'm trying to write a zero-shot prompt for a language model. The input of this model is in the format of:
"<prompt> <question> Answer:"

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. Do not be too long.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

error_example_template_alpha2 = """
<{index}> Question: {question}
The correct answer is: 
{gt_answer}
The correct result is: {label}.

The predicted answer is: 
{response}
The predicted result is {prediction}.
"""

'''
prompt alpha3
'''
gradient_prompt_tempelate_alpha3 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<prompt> <question> Answer:"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

Compare the correct and predicted answers above. Explain the reason why the prompt could have gotten these examples wrong.
""".strip()

optimize_prompt_tempelate_alpha3 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<prompt> <question> Answer:"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. Do not be too long.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

error_example_template_alpha3 = """
<{index}> Question: {question}
The correct answer is: 
{gt_answer}
The correct result is: {label}.

The predicted answer is: 
{response}
The predicted result is {prediction}.
"""

'''
prompt alpha4
'''
optimize_prompt_tempelate_alpha4 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<prompt> <question> Answer:"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 400 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

'''
prompt alpha5
'''
optimize_prompt_tempelate_alpha5 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<prompt> <question> Answer:"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 400 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

'''
prompt alpha6
'''
optimize_prompt_tempelate_alpha6 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<prompt> <question> Answer:"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt should be concise and the new prompt's length should be shorter than 100 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

'''
prompt alpha7
'''
optimize_prompt_tempelate_alpha7 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<prompt> <question> Answer:"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 100 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_alpha10 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<prompt> <question> Answer:"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 400 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

gradient_prompt_tempelate_alpha4 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<question> Answer: <prompt>"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

Compare the correct and predicted answers above. Explain the reason why the prompt could have gotten these examples wrong.
""".strip()

optimize_prompt_tempelate_alpha8 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<question> Answer: <prompt>"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 100 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()



optimize_prompt_tempelate_alpha9 = """
I'm trying to write a zero-shot prompt for a language model. 
The input of this model is in the format of: "<question> Answer: <prompt>"
I will take the last number in the model's response as the predicted answer.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()


gradient_prompt_tempelate_alpha5 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

Compare the correct and predicted answers above. Explain the reason why the prompt could have gotten these examples wrong.
""".strip()

optimize_prompt_tempelate_alpha11 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()


gradient_prompt_tempelate_alpha6 = """
I'm trying to write a prompt for a language model. The format of the model's input is {forward_format}.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

Compare the correct and predicted answers above. Explain the reason why the prompt could have gotten these examples wrong.
""".strip()

optimize_prompt_tempelate_alpha12 = """
I'm trying to write a prompt for a language model. The format of the model's input is {forward_format}.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that 
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_alpha13 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples the problem with this prompt is that:
{gradient}

Based on the above information, I try to write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 100 words.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

error_example_template_alpha4 = """
<{index}> 
Question and current prompt: 
{question}
The correct answer is: 
{gt_answer}
The correct result is: {label}.

The predicted answer is: 
{response}
The predicted result is {prediction}.
"""

error_example_template_alpha5 = """
<{index}> 
Question: 
{question}
Prompt:
{prompt}

The correct answer is: 
{gt_answer}
The correct result is: {label}.

The predicted wrong answer is: 
{response}
The predicted wrong result is {prediction}.
"""

gradient_prompt_tempelate_alpha6 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

Carefully examine the wrong answers, and by analyzing the role of the prompt in generating wrong answers, provide a comprehensive explanation of why the prompt can lead to these wrong answers.
""".strip()

gradient_prompt_tempelate_alpha7 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

Carefully examine each wrong answer, analyze why the prompt leads to the wrong answer, and how can the prompt be modified to generate the correct answer. Provide comprehensive and detailed explanations of why the prompt can lead to these wrong answers and suggestions to modify this prompt.
""".strip()

optimize_prompt_tempelate_alpha14 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problem with this prompt is that:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.
3. The new prompts solve the current prompt's problem and can guide the model to the correct answer.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

gradient_prompt_tempelate_alpha8 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

For each example, carefully examine each question and wrong answer, analyze why the prompt leads to the wrong answer in detail, and what aspect the prompt should focus on to help the model get to the correct answer. List comprehensive explanations of why the prompt can lead to these wrong answers and diverse aspects to modify this prompt to help the model answer these questions correctly.
""".strip()

gradient_prompt_tempelate_alpha9 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive reasons why the prompt leads to the wrong answer. Based on all the information, list comprehensive and different explanations of why the prompt can lead to these wrong answers.
""".strip()

gradient_prompt_tempelate_alpha10 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
""".strip()

optimize_prompt_tempelate_alpha15 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problem with this prompt is that:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.
3. The new prompts should retain the good aspects of the current prompt and solve the current prompt's problem.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

correct_example_template_alpha0 = """
<{index}> 
Question: 
{question}
Prompt:
{prompt}

The correct answer is: 
{gt_answer}
The correct result is: {label}.

The predicted correct answer is: 
{response}
The predicted correct result is {prediction}.
"""

correct_gradient_prompt_tempelate_alpha0 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

This prompt gets the following examples correct:
{correct_string}

For each correct example, carefully examine each question and correct answer step by step, provide comprehensive reasons why the prompt leads to the correct answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
""".strip()

optimize_prompt_tempelate_alpha16 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

This prompt help the model correctly answer the following examples:
{correct_string}

Based on these correct examples, the advantages of this prompt are:
{correct_gradient}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correctly answer all these examples and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.
3. The new prompts should retain the advantages of the current prompt and solve the current prompt's problem.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

correct_gradient_prompt_tempelate_alpha1 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

This prompt help the model correctly answer the following examples:
{correct_string}

List the advantages of the current prompt:
""".strip()

optimize_prompt_tempelate_alpha17 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

This prompt help the model correctly answer the following examples:
{correct_string}

But this prompt gets the following examples wrong:
{error_str}

Based on the correct examples, the advantages of this prompt are:
{correct_gradient}

Based on the wrong examples, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correctly answer all these examples and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.
3. The new prompts should retain the advantages of the current prompt and solve the current prompt's problem.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_alpha18 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problem with this prompt is that:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.
3. The new prompts should be based on the current prompt, retain the advantages of the current prompt, and solve the current prompt's problem.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

'''
Forward prompts
'''
error_example_template_alpha6 = """
<{index}> 
Prompt:
{prompt}
Question: 
{question}

The correct answer is: 
{gt_answer}
The correct result is: {label}.

The predicted wrong answer is: 
{response}
The predicted wrong result is {prediction}.
"""

optimize_prompt_tempelate_alpha19 = """
I'm trying to write a zero-shot prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problem with this prompt is that:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 100 words.
3. The new prompts should be based on the current prompt, retain the advantages of the current prompt, and solve the current prompt's problem.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

error_example_template_alpha7 = """
<{index}> 
{question}

The correct result is: {label}.

The predicted wrong answer is: 
{response}
The predicted wrong result is {prediction}.
"""

optimize_prompt_tempelate_alpha20 = """
I need to create an improved version of a prompt for a language model.

Original Prompt:
{cur_prompt}

Issues observed:
{error_str}

The main problem identified:
{gradient}

Please consider these guidelines:
1. Exclude specific details from the original questions.
2. Keep the new prompt under {prompt_length_limit} words.
3. Build upon the original, highlighting its strengths and addressing its flaws.

I've made {steps_per_gradient} variations. They start with <START> and end with <END>. Here they are:
""".strip()

gradient_prompt_tempelate_alpha11 = """
I need to refine a prompt for a language model.

Original Prompt:
{cur_prompt}

Issues observed:
{error_string}

Please analyze each problematic example in detail and explain why the prompt resulted in incorrect answers. Lastly, provide a summarized list of improvements to enhance the prompt's effectiveness.
""".strip()

error_example_template_alpha8 = """
Entry <{index}>:

Question:
{question}

Correct Answer:
{label}

Given Incorrect Answer:
{response}

Wrong Prediction:
{prediction}
"""


error_example_template_beta0 = """
<{index}> 
The model's input is:
{question}

The correct answer is: {label}.

The model's response is: 
{response}
The predicted wrong answer is {prediction}.
"""

gradient_prompt_tempelate_beta0 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt fails to help the model answer the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt to solve the whole task.
""".strip()

optimize_prompt_tempelate_beta0 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts based on the current prompt that solve those problems and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than {prompt_length_limit} words.
3. Retain the good aspects of the current prompt.
4. The new prompt should provide practical guidance to help the prompt solve the whole task.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()


error_example_template_beta1 = """
<{index}> 
{question}

The correct answer is: {label}.

The model's response is: 
{response}
The predicted wrong answer is {prediction}.
"""

gradient_prompt_tempelate_beta1 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
""".strip()

optimize_prompt_tempelate_beta1 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problem with this prompt is that:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 100 words.
3. The new prompts should retain the advantages of the current prompt, and solve the current prompt's problem.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta2 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.
3. The new prompts should retain the advantages of the current prompt, and solve the current prompt's problem.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta3 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 200 words.
3. The new prompts should retain the most helpful parts of the current prompt.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

error_example_template_beta2 = """
<{index}> 
The model's input is:
{question}

The correct label is: {label}.

The model's response is: 
{response}

The model's prediction is {prediction}.
"""

optimize_prompt_tempelate_beta4 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these examples, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can solve those problems and follow these constraints:
1. Do not involve the specific information in the above examples.
2. The new prompt's length should be shorter than 200 words.
3. The new prompts should keep the most helpful parts of the current prompt.

I wrote {steps_per_gradient} different improved prompts based on the current prompt. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta5 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 300 words.
3. The new prompts should retain the key parts of the current prompt.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

error_example_template_beta3 = """
<{index}>
The model's input is:
{question}
The correct label is: {label}.

The model's response is:
{response}
The model's prediction is {prediction}.
"""

correct_example_template_beta0 = """
<{index}>
The model's input is:
{question}
The correct label is: {label}.

The model's response is:
{response}
The model's prediction is {prediction}.
"""

gradient_prompt_tempelate_beta2 = """
I'm trying to write a prompt for a language model.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
""".strip()

optimize_prompt_tempelate_beta6 = """
I'm trying to write a prompt for a language model. 

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 300 words.
3. The new prompts should combine the key parts of the current prompt.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta7 = """
I'm trying to write a prompt for a language model.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 300 words.
3. The new prompts should retain the most helpful parts of the current prompt.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta8 = """
I'm trying to write a prompt for a language model.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than 300 words.
3. The new prompts should retain the most important parts of the current prompt.

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta9 = """
I'm trying to write a prompt for a language model.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompts should retain the most important parts of the current prompt ({cur_prompt}).

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta10 = """
I'm trying to write a prompt for a language model.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than {prompt_length_limit} words.
3. The new prompts should retain the most important parts of the current prompt ({cur_prompt}).

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta11 = """
I'm trying to write a prompt for a language model.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can solve these problems and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt should be appended in front of the current prompt ({cur_prompt}).

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()

optimize_prompt_tempelate_beta12 = """
I'm trying to write a prompt for a language model.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

Based on these errors, the problems with this prompt are:
{gradient}

Based on the above information, I write new prompts that can correct those errors and follows these constraints:
1. Do not involve the specific information in the above questions.
2. The new prompt's length should be shorter than {prompt_length_limit} words.
3. The new prompts should retain the main parts of the current prompt ({cur_prompt}).

I wrote {steps_per_gradient} different improved prompts. Each prompt is wrapped with <START> and <END>. The new prompts are:
""".strip()



trajectory_sample_template_gamma0 = """
Given the following prompts:
{trajectory_prompts}

Each of the prompt above evolves from its former prompts. Based on the prompts above, please create a new prompt. Enclose it with <START> and <END>.
""".strip()

gradient_prompt_tempelate_gamma0 = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_string}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
""".strip()

optimize_prompt_tempelate_gamma0 = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{gradient}

There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
{trajectory_prompts}

Based on the above information, please write 1 new prompt following these guidelines:
1. The new prompt should solve the current prompt's problems.
2. The new prompt should consider the list of prompts and evolve based on the current prompt.
3. The new prompt should be wrapped with <START> and <END>.

The new prompts is:
""".strip()

error_example_template_gamma0 = """
<{index}> 
The model's input is:
{question}

The correct label is: {label}.

The model's response is: 
{response}

The model's prediction is {prediction}.
"""

error_example_template_gamma1 = """
<{index}> 
The model's input is:
{question}

The model's response is: 
{response}

The correct label is: {label}
The model's prediction is: {prediction}.
"""

optimize_prompt_tempelate_gamma1 = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

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

optimize_prompt_tempelate_gamma2 = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_str}

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