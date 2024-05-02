from torch.cuda import is_available 
from torch import no_grad
from transformers import AutoModel, AutoTokenizer

def load_LLM(model_id):
    # use GPU if available, on mac can use MPS
    device = "cuda" if is_available() else "cpu"
    print(f'device: {device}')

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return model,tokenizer,  device


def embed(docs: list[str],tokenizer= None,model= None,device = None) -> list[list[float]]:
    docs = [f"passage: {d}" for d in docs]
    # tokenize
    tokens = tokenizer(
        docs, padding=True, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    with no_grad():
        # process with model for token-level embeddings
        out = model(**tokens)
        # mask padding tokens
        last_hidden = out.last_hidden_state.masked_fill(
            ~tokens["attention_mask"][..., None].bool(), 0.0
        )
        # create mean pooled embeddings
        doc_embeds = last_hidden.sum(dim=1) / \
            tokens["attention_mask"].sum(dim=1)[..., None]
    return doc_embeds.cpu().numpy().tolist()