from transformers import Trainer
from torch.nn import functional as F
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, self.model.config.num_labels),
            labels,
            weight=self.class_weights,
        )
        return (loss, outputs) if return_outputs else loss

    def set_class_weights(self, weights):
        self.class_weights = weights

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro"
        )
        acc = accuracy_score(labels, preds)
        c_mat = confusion_matrix(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": c_mat,
        }


def hyperspace(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16]
        ),
    }
