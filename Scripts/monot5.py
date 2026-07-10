import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class MonoT5:
    def __init__(self, model_name_or_path, token_false="▁no", token_true="▁yes"):
        # Attempt to load locally first (for offline nodes), then fall back to Hub
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            
        self.token_false = token_false
        self.token_true = token_true
        
        self.true_id = self.tokenizer.convert_tokens_to_ids(token_true)
        self.false_id = self.tokenizer.convert_tokens_to_ids(token_false)

    def predict(self, sentence_pairs, batch_size=32):
        scores = []
        device = next(self.model.parameters()).device
        self.model.eval()
        
        for i in range(0, len(sentence_pairs), batch_size):
            batch = sentence_pairs[i:i + batch_size]
            # Format: "Query: {q} Document: {d} Relevant:"
            inputs_text = [f"Query: {q} Document: {d} Relevant:" for q, d in batch]
            inputs = self.tokenizer(
                inputs_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                decoder_input_ids = torch.zeros((len(batch), 1), dtype=torch.long, device=device)
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                logits = outputs.logits[:, 0, :]
                true_logits = logits[:, self.true_id]
                false_logits = logits[:, self.false_id]
                batch_scores = (true_logits - false_logits).cpu().tolist()
                scores.extend(batch_scores)
                
        return scores
