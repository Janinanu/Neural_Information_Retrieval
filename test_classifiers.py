import torch

def test_multiclass_model(model, test_loader, num_examples, long):
    """
    Computes test acc and loss of multiclass model
    :param model: loaded model
    :param test_loader: test documents
    :param num_examples: number of test documents
    :param long: datatype (cuda/cpu)
    :return: test acc and loss
    """
    loss_func = torch.nn.CrossEntropyLoss()
    sum_loss = 0
    sum_correct = 0

    for batch_id, (texts, labels) in enumerate(test_loader):
        _, hidden_scores = model(texts.type(long))
        labels = labels.type(long)

        cur_loss = loss_func(hidden_scores, labels)
        sum_loss += cur_loss.data

        _, prediction = hidden_scores.max(dim=-1)  # batchsize x 1
        sum_correct += sum([1 for i in range(len(labels)) if labels[i] == prediction[i]])

    test_loss = sum_loss / num_examples
    test_accuracy = sum_correct / num_examples

    return test_loss.item(), test_accuracy


def test_pair_cel_model(model, test_loader, num_examples, long, float):
    """
    Computes test loss for document pair model
    :param model: loaded model
    :param test_loader: test examples
    :param num_examples: number of test examples
    :param long: datatype (cuda/cpu)
    :param float: datatype (cuda/cpu)
    :return: test loss
    """

    loss_func = torch.nn.CosineEmbeddingLoss()
    sum_loss = 0

    for batch_id, (text_a, text_b, label) in enumerate(test_loader):
        last_hidden_a = model(text_a.type(long))
        last_hidden_b = model(text_b.type(long))

        for i in range(len(label)):
            if label[i] == 0.0:
                label[i] = -1.0

        cur_loss = loss_func(last_hidden_a, last_hidden_b, label.type(float))
        sum_loss += cur_loss.data

    test_loss = sum_loss / num_examples

    return test_loss.item()


def test_triplet_model(model, test_loader, num_examples, long, float, confidence):
    """
    Computes test loss for document triplet model
    :param model: loaded model
    :param test_loader: test examples
    :param num_examples: number of test examples
    :param long: datatype (cuda/cpu)
    :param float: datatype (cuda/cpu)
    :param confidence: if diff > threshold, count as correct
    :return: test loss and acc
    """

    loss = torch.nn.BCEWithLogitsLoss()
    sum_loss = 0
    sum_correct = 0

    for batch_id, (text_a, text_b, text_c, targets) in enumerate(test_loader):
        score_1, score_2 = model(text_a.type(long), text_b.type(long), text_c.type(long))
        diff = score_1 - score_2

        cur_loss = loss(diff, targets.type(float))
        sum_loss += cur_loss.data

        sigmoids = torch.sigmoid(diff).detach().cpu().numpy()
        sum_correct += sum([1 for d in sigmoids if d >= confidence])

    test_loss = sum_loss / num_examples
    test_accuracy = sum_correct / num_examples

    return test_loss.item(), test_accuracy
