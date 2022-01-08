import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def get_f1(preds, labels):
    preds_ = preds.argmax(1).cpu().detach().numpy()
    labels_ = labels.cpu().detach().numpy()
    return f1_score(labels_, preds_, average="micro")


def accuracy(preds, labels):
    preds_ = preds.argmax(1).cpu().detach().numpy()
    labels_ = labels.cpu().detach().numpy()
    return accuracy_score(labels_, preds_)


def get_result_lists(preds, labels):
    preds_ = preds.argmax(1).cpu().detach().numpy()
    labels_ = labels.cpu().detach().numpy()
    return labels_.tolist(), preds_.tolist()



def get_evaluation_metrics(model, dataloader, device):
    model.eval()
    accumulated_preds = []
    accumulated_labels = []
    with torch.no_grad():
        for _, i in enumerate(dataloader):
            input_ids = i["input_ids"].to(device)
            attention_mask = i["attention_mask"].to(device)
            labels = i["labels"].to(device)
            output = model(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            labels, preds = get_result_lists(output[1], labels)
            accumulated_labels += labels
            accumulated_preds += preds
    acc_f1_score = f1_score(accumulated_labels, accumulated_preds, average="micro")
    acc_conf_mat = confusion_matrix(accumulated_labels, accumulated_preds)
    return acc_f1_score, acc_conf_mat
