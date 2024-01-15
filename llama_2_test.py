#llama 2 interface:
#https://huggingface.co/meta-llama
#https://huggingface.co/blog/llama2



from transformers import AutoTokenizer
import transformers
import torch
from time import time

model = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    start = time()
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        'they ',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,
    )
    print(sequences)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    end = time()
    print('Time taken: {0:.2f}'.format(int(end-start)))




if __name__ == "__main__":
    main()  