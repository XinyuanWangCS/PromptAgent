# PromptAgent

PromptAgent is a prompt optimization method that autonomously crafts prompts equivalent in quality to those handcrafted by experts, i.e., expert-level prompts. ([arXiv(https://arxiv.org/abs/2310.16427)])

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

## Run MCTS agent

The following command run PromptAgent to craft an expert prompt for a BIG-bench task, [`penguins_in_a_table`](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/penguins_in_a_table). 


```bash
python src/main.py --task_name bigbench  --search_algo mcts --pred_model gpt-3.5-turbo --log_dir logs/ --post_instruction False --init_prompt "Answer questions about a table of penguins and their attributes." --train_shuffle True --batch_size 5 --expand_width 3 --train_size 70 --eval_size 70 --test_size 79 --iteration_num 12 --depth_limit 8 --data_dir datasets/penguins_in_a_table.json --api_key 
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

The initial query from the BIG-bench dataset is `Answer questions about a table of penguins and their attributes.`. Starting with such an ordinary prompt, PromptAgent will strategically sample model errors (from the base model), generate error feedbacks (actions), simulate future rewards, and search for high-reward paths leadning to expert prompts. The optimized prompt for `penguins_in_a_table` will look like this (exact results may vary as this is not deterministic):
```
As you delve into a dataset of penguins, assess essential attributes like names, ages, and gender. Decode the significance of each attribute in
the context of every penguin while keeping in mind that the dataset may be modified, including addition or removal of penguins. When
such modifications are made, immediately revise your understanding, redo your computations, and ensure that your subsequent calculations consider these changes. The crux of your task is to identify relationships and patterns within the attributes, giving special attention to the names and ages of the penguins.

For complex tasks, break them down into manageable chunks ensuring no essential detail is missed. When a change is made to the dataset, recompute your values taking into consideration these changes, paying extra attention to cumulative computations. Ensure that your understanding of ’more than’, ’less than’, and ’equal to’ is precise and that you correctly interpret these in context of the question.

...
```

After finishing the optimization, we can run `test.py` to the prompt performance with the following command:
```bash
python src/test.py --task_name bigbench --exp_name "20230913_1609-bigbench_geometric_shapes-algo_mcts-batch_2-train_150-eval_5-test_5" --eval_prompt "your prompt" --api_key "your_api" --data_dir "bigbench json path"
```

## How to add a new task?

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