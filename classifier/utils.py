import json

import torch
from torch.utils.data import Dataset


from sklearn.metrics import accuracy_score


class ICCVRouterDataset(Dataset):
    def __init__(self, path_to_json, tokenizer, task="question", max_length=None):
        super().__init__()

        # Task mappings
        self.task_to_id = {
            "left_right": 0,  # Binary classification
            "count": 1,  # Regression
            "distance": 2,  # Regression
            "mcq": 3,  # Regression (choice index)
        }

        with open(path_to_json) as f:
            self.data = json.load(f)

        if task == "question":
            self.texts = [x["conversations"][0]["value"] for x in self.data]

        elif task == "answer":
            self.texts = [x["conversations"][1]["value"] for x in self.data]

        self.labels = [self.task_to_id[x["category"]] for x in self.data]

        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = {
            k: torch.tensor(self.encodings[k][idx])
            for k in ("input_ids", "attention_mask")
        }
        item["labels"] = torch.tensor(self.labels[idx]).long()

        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}
