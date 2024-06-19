import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


# 验证模型
def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth_non0 = test_truth[non_zeros] > 0
    binary_preds_non0 = test_preds[non_zeros] > 0
    f_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')
    acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)

    binary_truth_has0 = test_truth >= 0
    binary_preds_has0 = test_preds >= 0
    acc_2 = accuracy_score(binary_truth_has0, binary_preds_has0)
    f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')

    mult_a7 = mult_a7*100
    acc1 = np.round(acc_2, 4)*100
    acc2 = np.round(acc_2_non0, 4)*100
    f1_1 = np.round(f_score, 4)*100
    f1_2 = np.round(f_score_non0, 4)*100
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc_7: ", mult_a7)
    print("mult_acc_5: ", mult_a5)
    print("Accuracy all/non0: {}/{}".format(acc1, acc2))
    print("F1 score all/non0: {}/{} over {}/{}".format(f1_1, f1_2,
                                                       binary_truth_has0.shape[0], binary_truth_non0.shape[0]))

    print("-" * 50)
    to_exl = [mae, corr, mult_a7, acc1, acc2, f1_1, f1_2]
    return {'mae': mae, 'corr': corr, 'mult': mult_a7, 'f1': f_score, 'acc2': acc_2, 'to_exl': to_exl}


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_humor(results, truths, exclude_zero=False):
    results = results.cpu().detach().numpy()
    truths = truths.cpu().detach().numpy()

    test_preds = np.argmax(results, 1)
    test_truth = truths

    print("Confusion Matrix (pos/neg) :")
    print(confusion_matrix(test_truth, test_preds))
    print("Classification Report (pos/neg) :")
    print(classification_report(test_truth, test_preds, digits=5))
    print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))


# def eval_mosi(results, truths, single=-1):
#     emos = ["Neutral", "Happy", "Sad", "Angry"]
#     test_preds = results.view(-1, 7, 2).cpu().detach().numpy()
#     test_truth = truths.view(-1, 7).cpu().detach().numpy()
#
#     for emo_ind in range(4):
#         print(f"{emos[emo_ind]}: ")
#         test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
#         test_truth_i = test_truth[:, emo_ind]
#         f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
#         acc = accuracy_score(test_truth_i, test_preds_i)
#         print("  - F1 Score: ", f1)
#         print("  - Accuracy: ", acc)


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()

        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
            test_truth_i = test_truth[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds, axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)


def eval_humor(results, truths, exclude_zero=False):
    results = results.cpu().detach().numpy()
    truths = truths.cpu().detach().numpy()

    test_preds = np.argmax(results, 1)
    test_truth = truths

    print("Confusion Matrix (pos/neg) :")
    print(confusion_matrix(test_truth, test_preds))
    print("Classification Report (pos/neg) :")
    print(classification_report(test_truth, test_preds, digits=5))
    acc_2=accuracy_score(test_truth, test_preds)
    print("Accuracy (pos/neg) ", acc_2)
    return {'acc2': acc_2}