forward_generation_tempelate = """
I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs.\nHere are the input-output pairs:
{examples}
The instruction was:
""".strip()

example_tempelate = "Input: {input}\nOutput: {output}\n"

resample_tempelate = """
Generate a variation of the following instruction while keeping the semantic meaning.
Input: {instruction}\nOutput:
""".strip()

