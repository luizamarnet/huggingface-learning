import re
from transformers import pipeline
import random 
import argparse

unmasker = pipeline('fill-mask', model='bert-base-cased', trust_remote_code=True)
#output = unmasker("Hello! My name is [MASK] and I'm from [MASK], so I speak [MASK].", top_k=2) # , top_k=2
#print(output)

def _clean(s: str) -> str:
    s = s.replace("[CLS] ", "").replace(" [SEP]", "")
    s = s.replace(" ' ", "'").replace(" n't", "n't")
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return s

def _pick(cands, temperature: float = 0.0):
    """Pick one candidate dict from a list of {'sequence','score',...}."""
    if not cands:
        raise ValueError("No token predicted.")
    # sampling
    if temperature <= 0:
        return cands[0]
    weights = [max(1e-17, c["score"]) ** (1.0 / temperature) for c in cands]
    total = sum(weights)
    weights = [w / total for w in weights]
    return random.choices(cands, weights=weights, k=1)[0]

def fill_all_masks(text: str = "Hello! My name is [MASK].", n_sentences: int = 1, temperature: float = 0.0) -> str:
    
    cur = [text for _ in range(n_sentences)]

    
    for i in range(n_sentences):

        
        # Fill one mask at a time using the top prediction for the FIRST remaining [MASK]
        while "[MASK]" in cur[i]:
            preds = unmasker(cur[i], top_k=1000)
            if isinstance(preds[0], list):
                choices = [pred for pred in preds[0]]
            else:
                choices = [pred for pred in preds]
            choice = _pick(choices, temperature=temperature)
            cur[i] = choice["sequence"]
            cur[i] = _clean(cur[i])
    return cur

'''text = "Hello! My name is [MASK]." # and I'm from [MASK], so I speak [MASK]."
print("\n\n")
output = fill_all_masks(text, n_sentences=5, temperature=0.8)
for sentence in output:
    print(sentence)'''

# --- main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill masks in text using Hugging Face BERT.")
    parser.add_argument("--text", type=str, required=True, help="Input text with [MASK] tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy).")
    parser.add_argument("--n_sentences", type=int, default=1, help="Number of sentences to generate.")

    args = parser.parse_args()

    outputs = fill_all_masks(args.text, n_sentences=args.n_sentences, temperature=args.temperature)
    for sent in outputs:
        print(sent)