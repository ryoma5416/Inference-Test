import os
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch


STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']
class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""

    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def get_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float16,
        # torch_dtype=torch.float32,
        # Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes
        # attn_implementation="flash_attention_2"
    )
    return model, tokenizer


def answer_prompt(question):
    prompt = (
        f"Answer the following question as briefly as possible.\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt


def generic_inference(model, tokenizer, prompt, stop_sequences=None):


    input_x = prompt + '\n'
    input_ids = tokenizer.encode(input_x, return_tensors="pt").to(model.device)
    if stop_sequences is not None:
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=stop_sequences + [tokenizer.eos_token],
            initial_length=len(input_ids[0]),
            tokenizer=tokenizer)])
    else:
        stopping_criteria = None

    output_token = model.generate(input_ids,
                                  do_sample=True,
                                  stopping_criteria=stopping_criteria,
                                  pad_token_id=tokenizer.eos_token_id,)
    output_text = tokenizer.decode(output_token[0][len(input_ids[0]):-1], skip_special_tokens=True)
    # print(output_text)

    return output_text


if __name__ == "__main__":
    # load model
    model_path = r'D:\PycharmProjects\LLM_Inference_test\models\Meta-Llama-3.2-3B-Instruct-hf'
    model, tokenizer = get_model_and_tokenizer(model_path)

    # load datasets
    # trivia_datasets = load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']  # 線上載入
    trivia_datasets = Dataset.from_parquet(r'datasets\TriviaQA-in-SQuAD-format.parquet')  # 本地載入
    # print(trivia_datasets)

    for i in range(5):
        print('-----------------------------------------------------------------------------')
        print('question :', trivia_datasets[i]['question'])
        print('answer :', trivia_datasets[i]['answers']['text'][0])
        # print('-----------------------------------------------------------------------------')
        prompt = answer_prompt(trivia_datasets[i]['question'])
        print('prompt :\n', prompt)
        print('-----------------------------------------------------------------------------')
        model_pred = generic_inference(model, tokenizer, prompt, stop_sequences=STOP_SEQUENCES)
        print('model_pred :\n', model_pred)
        print('-----------------------------------------------------------------------------')








