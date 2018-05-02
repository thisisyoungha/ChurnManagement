from utils.datapreprocess import Preprocess
from utils.extractor import Extractor
import numpy as np


def examine(input_file, data_path, epoch=10, batch_size=100, test_size=0.3, rotation=5, x_offset=1, class_n=1):
    pre = Preprocess()
    pre.class_n = class_n
    pre.data_path = data_path
    pre.x_offset = x_offset
    pre.verbose = 0

    X, Y = pre.perP_input(input_file)
    acc = []
    precision = []
    recall = []
    from utils.model import Churn_model
    input_dim = X.shape[1] - x_offset
    final_output_dim = class_n
    for i in range(rotation):
        print('%d번째 rotation' % (i+1))
        X_train, X_test, Y_train, Y_test = pre.split(X, Y, test_size=test_size, random_state=i)
        churn = Churn_model(input_dim=input_dim, final_output_dim=final_output_dim)
        churn.model_verbose = 0
        churn.epoch = epoch
        churn.batch_size = batch_size
        churn.train(X_train, X_test, Y_train, Y_test, x_offset)
        p_train = churn.predict(X_train, x_offset=x_offset)
        del X_train
        p_test = churn.predict(X_test, x_offset=x_offset)
        extractor = Extractor()
        extractor.x_offset = x_offset

        extractor.thresholding(Y_train, p_train)
        del p_train
        extractor.selected_threshold = extractor.roc_threshold  # roc로 계산
        extractor.fit(Y_test, p_test)
        acc.append((extractor.tp + extractor.tn) / (extractor.tp + extractor.tn + extractor.fp + extractor.fn))
        precision.append(extractor.precision)
        recall.append(extractor.recall)
        from keras import backend as K
        K.clear_session()
    print('mean of accuracy: %1.5f\n' % np.mean(acc),
          'mean of precision: %1.5f\n' % np.mean(precision),
          'mean of recall: %1.5f\n' % np.mean(recall))


def eval_performance(result_path, save=False, load=False, view_plot=False, make_prob=False):
    import pandas as pd
    import os
    # if base != 'acc' and base != 'roc':
    #     raise TypeError("base는 'acc'와 'roc' 둘 중 하나만 선택 가능")

    thresholds = pd.read_csv(os.path.join(result_path, 'results.txt'), sep='\t', header=0)
    threshold = thresholds['cutoff'].values

    test_score = pd.read_csv(os.path.join(result_path, 'test.txt'), sep='\t', header=0)
    pred_score = pd.read_csv(os.path.join(result_path, 'pred.txt'), sep='\t', header=0)

    test_score = test_score['p_score'].sort_values().values
    pred_score = pred_score['p_score'].sort_values().values

    def to_prob(x, t):  # 확률화시키는 transformation
        if x < t:
            return x / (2 * t)
        else:
            return 1 - (1 - x) / (2 * (1 - t))

    if make_prob:
        test_score = np.array(list(map(lambda x: to_prob(x, threshold), test_score))).reshape(-1)
        pred_score = np.array(list(map(lambda x: to_prob(x, threshold), pred_score))).reshape(-1)
        temp = 0.5
    else:
        temp = threshold

    if load:
        results = pd.read_csv(os.path.join(result_path, 'results_%s.txt' % ('prob' if make_prob else 'score')),
                              sep='\t', header=0)
        total_test = results['total_test']
        p_sum_test = results['p_sum_test']
        p_norm_test = results['p_norm_test']
        total_pred = results['total_pred']
        total_churn = results['total_churn']
        p_sum_pred = results['p_sum_pred']
        p_norm_pred = results['p_norm_pred']
    else:
        total_test = np.sum((test_score > threshold))
        p_sum_test = np.sum(test_score[(test_score > threshold)])
        p_norm_test = p_sum_test / total_test
        total_pred = np.sum((pred_score > threshold))
        total_churn = np.round(total_pred * p_norm_test)
        p_sum_pred = np.sum(pred_score[(pred_score > threshold)])
        p_norm_pred = p_sum_pred / total_pred
        for i in [0, 0.2, 0.4, 0.6, 0.8]:
            test = np.sum((test_score > i) * (test_score < (i + 0.2)))
            if test == 0:
                test = 1
            pred = np.sum((pred_score > i) * (pred_score < (i + 0.2)))
            if pred == 0:
                pred = 1

            p_s = np.sum(test_score[(test_score > i) * (test_score < (i + 0.2))])
            p_n = p_s / test
            p_s2 = np.sum(pred_score[(pred_score > i) * (pred_score < (i + 0.2))])
            p_n2 = p_s2 / pred
            if test == 1:
                test = 0
            if pred == 1:
                pred = 0
            total_test = np.append(total_test, test)
            p_sum_test = np.append(p_sum_test, p_s)
            p_norm_test = np.append(p_norm_test, p_n)
            p_sum_pred = np.append(p_sum_pred, p_s2)
            p_norm_pred = np.append(p_norm_pred, p_n2)
            total_pred = np.append(total_pred, pred)
            total_churn = np.append(total_churn, np.round(pred * p_n))
    if save:
        save_txt = np.concatenate((total_test.reshape(-1, 1), p_sum_test.reshape(-1, 1), p_norm_test.reshape(-1, 1),
                                   total_pred.reshape(-1, 1), total_churn.reshape(-1, 1), p_sum_pred.reshape(-1, 1),
                                   p_norm_test.reshape(-1, 1)), axis=1)
        np.savetxt(os.path.join(result_path, 'results_%s.txt' % ('prob' if make_prob else 'score')),
                   save_txt, delimiter='\t', fmt=['%d', '%.1f', '%1.3f', '%d', '%d', '%1.3f', '%1.3f'],
                   header="total_test\tp_sum_test\tp_norm_test\ttotal_pred\ttotal_churn\tp_sum_pred\tp_norm_pred",
                   comments='')
    if view_plot:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        num = list(range(len(test_score)))
        pos = num[np.argmin(np.abs(temp - test_score))]
        plt.plot(test_score, num, c='blue', alpha=0.5)
        plt.fill([0, temp, temp, 0], [0, 0, len(num), len(num)], c='m', alpha=0.1)
        plt.fill([1, temp, temp, 1], [0, 0, len(num), len(num)], c='b', alpha=0.1)
        plt.text(temp+0.005, pos, 'total: %d\nchurn: %d\n$\sum p$=%0.3f' %(total_test[0], p_sum_test[0], p_norm_test[0]),
                 verticalalignment='top', color='black', alpha=0.8, fontsize=8)
        for ind, i in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            plt.text(i, 0.05 * len(num),
                     '%.1f ~ %.1f\ntotal=%d\nchurn=%d\n$\sum p$=%0.3f'
                     % (i - 0.1, i + 0.1, total_test[ind + 1], p_sum_test[ind + 1], p_norm_test[ind + 1]),
                     fontsize=6, horizontalalignment='center', color=(1-i, .5, 1), alpha=0.5)
            plt.plot([i + 0.1, i + 0.1], [0, len(num)], ls="--", c='g', alpha=.3)

        plt.title("LRN-Test\ncut-off=%0.3f" % temp)
        plt.xlim([0, 1])
        plt.ylim([0, len(num)])

        plt.subplot(212)
        num = list(range(len(pred_score)))
        pos = num[np.argmin(np.abs(temp - pred_score))]
        plt.plot(pred_score, num, c='red', alpha=0.5)
        plt.fill([0, temp, temp, 0], [0, 0, len(num), len(num)], c='m', alpha=0.1)
        plt.fill([1, temp, temp, 1], [0, 0, len(num), len(num)], c='b', alpha=0.1)
        plt.text(temp+0.005, pos, 'total: %d\nchurn: %d\n$\sum$ p = %.3f'
                 % (total_pred[0], total_churn[0].astype(int), p_norm_pred[0]),
                 verticalalignment='top', color='black', alpha=0.8, fontsize=8)
        for ind, i in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            plt.text(i, 0.05*len(num), '%.1f ~ %.1f\ntotal=%d\np_churn=%d\n$\sum$ p =%.3f'
                     % (i - 0.1, i + 0.1, total_pred[ind+1], total_churn.astype(int)[ind+1], p_norm_pred[ind+1]),
                     fontsize=6, horizontalalignment='center', color=(1, .5, 1-i), alpha=0.5)
            plt.plot([i+0.1, i+0.1], [0, len(num)], ls="--", c='g', alpha=.3)
        plt.title("TRG")
        plt.xlim([0, 1])
        plt.ylim([-1, len(num)+10])
        if save:
            import pylab
            pylab.savefig(os.path.join(result_path, 'fig_%s.png' % ('prob' if make_prob else 'score')),
                          bbox_inches='tight')
        plt.show()
