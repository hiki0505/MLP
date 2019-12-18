from UFAZ.AIproject.functions import Functions as nn

class Results:
    def specificity(y_pred, y_true):
        tp, fp, fn, tn = nn.confusion_matrix(y_pred, y_true)
        return tn / (fp + tn)

    def f1_score(y_pred, y_true):
        tp, fp, fn, tn = nn.confusion_matrix(y_pred, y_true)
        return 2 * tp / (2 * tp + fp + fn)

    def precision(y_pred, y_true):
        tp, fp, fn, tn = nn.confusion_matrix(y_pred, y_true)
        return tp / (tp + fp)

    def recall(y_pred, y_true):
        tp, fp, fn, tn = nn.confusion_matrix(y_pred, y_true)
        return tp / (tp + fn)

    def accuracy(error):
        return (1 - error) * 100

    def cost(errors):
        return errors