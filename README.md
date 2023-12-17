<p align="center">
<img src="./images/Header.png" alt="Expert-level Prompting" title="Expert-level Prompting"/>
</p>

# PromptAgent

PromptAgent is a prompt optimization method that autonomously crafts prompts equivalent in quality to those handcrafted by experts, i.e., expert-level prompts. ([arXiv](https://arxiv.org/abs/2310.16427))

Unlike discovering magic/local prompt variants as existing prompt optimization methods, expert-level prompting is still an untapped area that solves challenging problems. And PromptAgent serves as a principled framework to study prompt optimization by unifying prompt sampling and rewarding. 

<p align="center">
<img src="./images/github_expert_prompt.png" alt="Expert-level Prompting" width="500" title="Expert-level Prompting"/>
</p>



## Installation

```bash
git clone https://github.com/XinyuanWangCS/PromptAgent.git
cd PromptAgent
conda create -n prompt_agent
conda activate prompt_agent
pip install -r requirements.txt
```


## Quick Start

The following command run PromptAgent to craft an expert prompt for a BIG-bench task, [penguins_in_a_table](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/penguins_in_a_table). The running could take some time depending on the inference speed of OpenAI APIs and size of datasets. 
```bash
python src/main.py --task_name bigbench --search_algo mcts --batch_size 5 --depth_limit 5 --train_size 70 --eval_size 50 --test_size 0 --seed 42 --train_shuffle True --iteration_num 10 --expand_width 3 --post_instruction False --pred_model gpt-3.5-turbo --optim_model gpt-4 --log_dir logs/ --data_dir datasets/penguins_in_a_table.json --init_prompt "Answer questions about a table of penguins and their attributes." --api_key "your_api_key"
```

`penguins_in_a_table` is an table understanding task to answer questions about animals contained in tables. An example from the original dataset looks like this:
```
Here is a table where the first line is a header and each subsequent line is a penguin:

name, age, height (cm), weight (kg)
Louis, 7, 50, 11
Bernard, 5, 80, 13
Vincent, 9, 60, 11
Gwen, 8, 70, 15

For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of
Bernard is 80 cm.

Which penguin is taller than the other ones? Answer:
```
Then, the expected result is `Bernard`.

The initial query from the BIG-bench dataset is `Answer questions about a table of penguins and their attributes.` Starting with such an ordinary prompt, PromptAgent will strategically sample model errors (from the base model), generate error feedbacks (actions), simulate future rewards, and search for high-reward paths leadning to expert prompts. The optimized prompt for `penguins_in_a_table` will look like this (exact results may vary as this is not deterministic):
```
As you delve into a dataset of penguins, assess essential attributes like names, ages, 
and gender. Decode the significance of each attribute in the context of every penguin 
while keeping in mind that the dataset may be modified, including addition or removal 
of penguins. When such modifications are made, immediately revise your understanding, 
redo your computations, and ensure that your subsequent calculations consider these 
changes. The crux of your task is to identify relationships and patterns within 
the attributes, giving special attention to the names and ages of the penguins.

For complex tasks, break them down into manageable chunks ensuring no essential detail 
is missed. When a change is made to the dataset, recompute your values taking into 
consideration these changes, paying extra attention to cumulative computations. Ensure 
that your understanding of ’more than’, ’less than’, and ’equal to’ is precise and 
that you correctly interpret these in context of the question.

...
```

After finishing the optimization, all the intermediate nodes and paths will be stored in a json file. We will keep the top-k reward nodes, the last node in the highest average reward path, and the highest reward node in the highest averaget reward path. In the paper, we use the last one as the selection strategy. 

### Test
We can run `test.py` to test any prompt performance with the following commands:  
Enter the prompt in the command line:
```bash
python src/test.py --task_name bigbench --eval_prompt "Answer questions about a table of penguins and their attributes." --prompt_file "prompt file path" --train_size 70 --eval_size 50 --test_size 79 --seed 42 --pred_model 'gpt-3.5-turbo' --data_dir "datasets/penguins_in_a_table.json" --api_key "your_api"
```
or   
Put prompt in a .txt file if the prompt is very long:
```bash
python src/test.py --task_name bigbench --prompt_file "prompt file path" --train_size 70 --eval_size 50 --test_size 79 --seed 42 --pred_model 'gpt-3.5-turbo' --data_dir "datasets/penguins_in_a_table.json" --api_key "your_api"
```

## How to add a new task?
Our current tasks includes selection question tasks and NER tasks. Adding new selection tasks is relatively easy. Please refer to the .py files in the tasks folder. First, create a new task.py file and a new CustomTask class. Then, there are several task-specific functions to be implemented in your customized task.py file: 
1. Load your dataset: We recommend spliting your dataset into "train" and "test" and storing them into json file.
2. Input formating: For selection questions, it is necessary to combine question and options before inputing into the pred_model.
3. Answer extraction: Extract the final answer from the model's response.   

After that, you can run PromptAgent on your customized dataset!

`TODO: We will extend the features to enable flexible training/testing pipeline with new tasks`


## Citations

If you find the paper and code useful, please kindly star this repo and cite the following paper. Feel free to contact <xiw136@ucsd.edu> and <zhenwang9102@gmail.com>, or open an issue if you have any questions. Thanks so much!

```bibtex
@article{wang2023promptagent,
  title={PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization},
  author={Wang, Xinyuan and Li, Chenxi and Wang, Zhen and Bai, Fan and Luo, Haotian and Zhang, Jiayou and Jojic, Nebojsa and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2310.16427},
  year={2023}
}
```