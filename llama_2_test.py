#llama 2 interface:
#https://huggingface.co/meta-llama
#https://huggingface.co/blog/llama2



from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float32, #change to float16 for faster inference on gpu
        device_map="auto",
    )

    sequences = pipeline(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    print(sequences)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        print('!')




if __name__ == "__main__":
    main()  