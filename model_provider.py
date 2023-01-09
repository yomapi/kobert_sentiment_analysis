import torch
import torch
import numpy as np
from bert_dataset import BERTDataset
from config import tok, vocab, batch_size, max_len, device


class SentimentAnalysisProvider:
    def __init__(self, model_path: str = "SentimentAnalysisKOBert.pt") -> None:
        self.model = torch.load(f"{model_path}")
        self.emotion_dict = {
            "0": "기쁨",
            "1": "불안",
            "2": "당황",
            "3": "슬픔",
            "4": "분노",
            "5": "상처",
        }

    def predict(self, sentence, model):
        dataset = [[sentence, "0"]]
        test = BERTDataset(dataset, 0, 1, tok, vocab, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, num_workers=2
        )
        print(test_dataloader)
        model.eval()
        answer = 0
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(
            test_dataloader
        ):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            for logits in out:
                logits = logits.detach().cpu().numpy()
                print(dict(zip(self.emotion_dict.values(), logits)))
            for logits in out:
                logits = logits.detach().cpu().numpy()
                answer = np.argmax(logits)
        return answer


sentiment_analysis_provider = SentimentAnalysisProvider()
