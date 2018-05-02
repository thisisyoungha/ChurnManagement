"""
extractor.py

threshold를 찾아내거나 distribution을 찾아내는 class 생성

Extractor
    thresholding: training set에 대해서 적절한 threshold 찾아냄
    fit: 결과 저장
    importance: 각 feature들의 영향력 출력
    print_score: fit을 통해 저장된 데이터 출력
    roc_view: roc curve 출력
    importance_view: 영향력 histogram으로 출력
    see_distribution: distribution 확인
    see_density: density 함수 형태로 출력
    see_density_pred: 결과값에 대한 density 함수 출력
"""
from sklearn import metrics
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


class Extractor:
    def __init__(self):
        # input
        self.save = False
        self.result_path = '.'
        self.data_path = '.'
        self.verbose = 0
        self.x_offset = 1
        self.num_view = 10,
        self.feature_index = 84
        self.delete_index = []
        self.delete_index_g = []
        self.delete_feature = False

        # output
        self.acc_threshold = 0.5
        self.roc_threshold = 0.5
        self.selected_threshold = self.roc_threshold
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0

    def thresholding(self, Y_train, p_train):
        fpr, tpr, threshold_list = metrics.roc_curve(Y_train, p_train)

        # tpr과 1-fpr의 차이가 0에 가까울 때 roc_threshold
        local_roc = np.abs(tpr - (1 - fpr))
        self.roc_threshold = threshold_list[np.argmin(local_roc, axis=0)]

        # acc가 최대가 되는 cut-off 찾기
        past_thresh = 0
        past_acc = 0
        current_thresh = 0
        current_acc = 0.001
        delta = 0.1
        direction = 1
        patient = 0
        while patient < 31:  # 30번 참고 넘으면 결정해버림
            # print(patient, direction, delta, past_thresh, current_thresh, past_acc, current_acc)
            if past_acc < current_acc:
                patient = 0
                past_thresh = current_thresh
                past_acc = current_acc
                current_thresh = current_thresh + delta * direction  # delta: 증분,
                current_acc = metrics.accuracy_score(Y_train[:100000], (p_train[:100000] > current_thresh) * 1)
            else:
                delta /= 2  # delta를 1/2씩 감소
                patient += 1
                current_thresh = past_thresh + delta * direction
                current_acc = metrics.accuracy_score(Y_train[:100000], (p_train[:100000] > current_thresh) * 1)
                if patient % 15 == 0:  # 15번 참으면 방향 바꿈
                    direction *= -1
                    delta *= 2 ** 15
                    current_thresh = past_thresh
                    current_acc = past_acc
        self.acc_threshold = current_thresh

    def fit(self, Y_test, p_test):
        # 전부 다 계산해서 class에 저장
        self.tn, self.fp, self.fn, self.tp = metrics.confusion_matrix(Y_test,
                                                                      (p_test > self.selected_threshold)).ravel()
        self.accuracy = metrics.accuracy_score(Y_test, (p_test > self.selected_threshold))
        self.precision = metrics.precision_score(Y_test, (p_test > self.selected_threshold))
        self.recall = metrics.recall_score(Y_test, (p_test > self.selected_threshold))
        self.f1 = metrics.f1_score(Y_test, (p_test > self.selected_threshold))

    def importance(self, input_data, output_data):
        # ExtraTree로 importance 계산하는 방법 채택
        from sklearn.ensemble import ExtraTreesClassifier
        importance_model = ExtraTreesClassifier()
        importance_model.fit(input_data, output_data)
        importances = importance_model.feature_importances_  # 전부 합하면 1이므로 score라고 생각

        from utils.datapreprocess import load_info
        feature_info = load_info(self.data_path, 'feature_info')
        group_info = load_info(self.data_path, 'feature_info_label')
        feature_names = feature_info['변수명'].values[self.x_offset:]
        if self.delete_feature:
            feature_names = np.delete(feature_names, self.delete_index, None)
        importance_list = pd.DataFrame(data={'feature': feature_names, 'importance': importances})

        # feature name 가져와서 매칭시킴
        group_names = group_info['group'].values[:]
        label = group_info['number'].values[:]
        if self.delete_feature:
            group_names = np.delete(group_names, self.delete_index_g, None)
            label = np.delete(label, self.delete_index_g, None)

        # group 별로 묶기
        accum_label = [label[0]]
        for i in range(1, len(label)):
            accum_label.append(label[i] + accum_label[i - 1])
        # group별 importances
        group_importances = [sum(importances[0:accum_label[0]])]
        for i in range(1, len(accum_label)):
            group_importances.append(sum(importances[accum_label[i - 1]:accum_label[i]]))
        group_importance_list = pd.DataFrame(data={'feature': group_names, 'importance': group_importances})
        return importance_list, group_importance_list

    def print_score(self, test_set, predict_set):
        # 결과치 출력
        print('=======================================================')
        print('cut-off value: %1.4f' % (self.selected_threshold))
        print('confusion matrix is...')
        print(metrics.confusion_matrix(test_set, (predict_set > self.selected_threshold) * 1))  # TP/ FP/ FN / TN
        print("tn: %d,\tfp: %d,\tfn: %d,\ttp: %d" % (self.tn, self.fp, self.fn, self.tp))
        print('acc: %1.4f,\tprecision(정확도): %1.4f\trecall(power): %1.4f\tf1-score:%1.4f'
              % (metrics.accuracy_score(test_set, (predict_set > self.selected_threshold) * 1),
                 self.precision, self.recall, self.f1))

    def roc_view(self, Y_test, p_test):
        # roc_curve 출력
        print('선택된 cut-off value: \nacc %1.4f, \nroc %1.4f' % (self.acc_threshold, self.roc_threshold))
        fpr, tpr, threshold_list = metrics.roc_curve(Y_test, p_test)  # ROC curve 구하는 시간 꽤 걸림
        # visualization: test set에 대한 ROC curve
        roc_auc = metrics.auc(fpr, tpr)
        plt.title("ROC")
        plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def importance_view(self, importance_list, name=''):
        # importance 그래프로 출력

        # TOP 10 출력
        top_ten = importance_list.sort_values(by=['importance'], ascending=False)[0:10]
        print('-------------------------------------------------------\n' +
              'Top 10 - ' + name + ' feature importances')
        print(top_ten)  # Order by로 10개 출력
        # 그래프로 함께 출력
        plt.figure()
        plt.title(name + " feature importances")
        plt.bar(np.linspace(0, 19, num=10), top_ten['importance'].values,
                color="r", align="center")
        plt.xticks(np.linspace(0, 19, num=10), top_ten['feature'].values, rotation=30)
        plt.xlim([-1, 20])
        plt.show()

        bottom_ten = importance_list.sort_values(by=['importance'], ascending=True)[0:self.num_view]
        print('-------------------------------------------------------\n' +
              'Bottom %d - ' % self.num_view + name + 'feature importances')
        print(bottom_ten)

    def final_sale(self, X_test, Y_test, p_test):
        # 구매주기와 관련된 feature들(84~87)에 대해서 이탈/안이탈 과 prediction 값을 비교
        if self.delete_feature:
            self.feature_index = 84 - np.sum(np.array(self.delete_index) < 84)
        up = []
        down = []
        count = []
        X_test = X_test[:, self.x_offset:]
        if X_test.shape[0] < 10000:
            point_size = 2
        elif X_test.shape[0] < 100000:
            point_size = 1
        else:
            point_size = 0.5

        plt.title("brief summary")
        plt.plot([-2, 11], [self.selected_threshold, self.selected_threshold], c='gray',
                 label='threshold: %1.2f' % self.selected_threshold, linewidth=2.0)

        final_sale = np.argmax(X_test[:, self.feature_index:(self.feature_index + 4)], axis=1)  # index화
        for i in range(4):  # 1~91, 92~182, 183~273, 274~364
            friends = np.concatenate(
                (X_test[final_sale == i, 0:1], p_test[final_sale == i, :]), axis=1)
            up.append(sum(friends[:, 1] > self.selected_threshold) / len(friends) * 100)
            down.append(sum(friends[:, 1] < self.selected_threshold) / len(friends) * 100)
            count.append(friends.shape[0])
            # blue: 실제 이탈안한 친구들 (Y_test == 0)
            random.seed(1)
            friends = np.concatenate(
                (X_test[(final_sale == i) * (np.transpose(Y_test) == 0)[0], 0:1],
                 p_test[(final_sale == i) * (np.transpose(Y_test) == 0)[0], :]
                 ), axis=1)
            plt.scatter(3 * i - 0.5 + np.random.randn(len(friends)) * 0.2, friends[:, 1], c='skyblue', s=point_size)
            # red: 실제로 이탈한 친구들 (Y_test == 1)
            random.seed(1)
            friends = np.concatenate(
                (X_test[(final_sale == i) * (np.transpose(Y_test) == 1)[0], 0:1],
                 p_test[(final_sale == i) * (np.transpose(Y_test) == 1)[0], :]
                 ), axis=1)
            plt.scatter(3 * i + 0.5 + np.random.randn(len(friends)) * 0.2, friends[:, 1], c='salmon', s=point_size)
        plt.legend(loc='lower right')
        plt.xlim([-2, 11])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('probability')
        plt.xlabel('classes')
        plt.xticks(np.arange(0, 12, step=3), ('class 1', 'class 2', 'class 3', 'class 4'))
        if self.verbose == 2:
            print('-------------------------------------------------------\n' +
                  'class\t:\t1\t2\t3\t4\n' +
                  'up\t:\t%3.1f\t%3.1f\t%3.1f\t%3.1f\n' % (up[0], up[1], up[2], up[3]) +
                  'down\t:\t%3.1f\t%3.1f\t%3.1f\t%3.1f\n' % (down[0], down[1], down[2], down[3]) +
                  'total\t:\t%d\t%d\t%d\t%d' % (count[0], count[1], count[2], count[3]))
        plt.show()

    def see_all(self, X_test, Y_test, p_test):
        # verbose에 따라 출력될 것들 정해짐
        if self.verbose >= 1:
            self.importance_list, self.group_importance_list = self.importance(
                X_test[:100000, self.x_offset:], ((p_test[:100000, :] > self.selected_threshold) * 1).ravel())
        if self.verbose == 1:
            self.print_score(Y_test, p_test)
            self.importance_view(self.importance_list, 'grouped')
        elif self.verbose == 2:
            self.print_score(Y_test, p_test)
            self.roc_view(Y_test, p_test)
            self.final_sale(X_test, Y_test, p_test)
            self.importance_view(self.importance_list, 'grouped')
            self.importance_view(self.importance_list, 'individual')
        elif self.verbose == 3:
            self.print_score(Y_test, p_test)
            self.importance_view(self.importance_list, 'grouped')
        if self.verbose >= 1:
            return self.importance_list, self.group_importance_list

    def save_output_new(self, name, X_test, Y_test, p_test, New_data, p_new):
        # 결과치 저장
        subresult_path = os.path.join(self.result_path, os.path.basename(name))
        if not os.path.exists(subresult_path):
            os.mkdir(subresult_path)
        results = [self.selected_threshold, self.precision, self.recall, self.tn, self.fp, self.fn, self.tp]
        # cut-off/ precision/ recall/ tn, fp, fn, tp/
        np.savetxt(
            os.path.join(subresult_path) + os.path.sep + 'results' + '.txt', [results],
            delimiter='\t', fmt=['%1.10f', '%1.10f', '%1.10f', '%d', '%d', '%d', '%d'],
            header="cutoff\taccuracy\tpower\ttn\tfp\tfn\ttp", comments='')

        result_test = np.concatenate(  # COMCSNO, 실제 정답, 예측값, 예측 score
            (X_test[:, 0:1], Y_test[:, :], (p_test > self.selected_threshold), p_test[:, :]), axis=1)
        result_pred = np.concatenate(
            (New_data[:, 0:1], (p_new > self.selected_threshold), p_new[:, :]), axis=1)
        # test set에 대한 result 저장
        np.savetxt(
            os.path.join(subresult_path) + os.path.sep + 'test' + '.txt', result_test,
            delimiter='\t', fmt=['%d', '%d', '%d', '%1.10f'], header="COMCSNO\tY\tY_hat\tp_score", comments='')
        # TRN에 대한 predict result 저장
        np.savetxt(
            os.path.join(subresult_path) + os.path.sep + 'pred' + '.txt', result_pred,
            delimiter='\t', fmt=['%d', '%d', '%1.10f'], header="COMCSNO\tY_hat\tp_score", comments='')

    def save_output(self, name, X_test, Y_test, p_test, New_data, p_new):
        # 결과치 저장
        subresult_path = os.path.join(self.result_path, os.path.basename(name))
        if not os.path.exists(subresult_path):
            os.mkdir(subresult_path)
        thresholds = [self.acc_threshold, self.roc_threshold]
        np.savetxt(
            os.path.join(subresult_path) + os.path.sep + 'threshold_info' + '.txt', [thresholds],
            delimiter='\t', fmt=['%1.10f', '%1.10f'], header="acc\troc", comments='')

        statistics = 'roc'
        result_test = np.concatenate(  # COMCSNO, 실제 정답, 예측값, 예측 score
            (X_test[:, 0:1], Y_test[:, :], (p_test > self.selected_threshold), p_test[:, :]), axis=1)
        result_pred = np.concatenate(
            (New_data[:, 0:1], (p_new > self.selected_threshold), p_new[:, :]), axis=1)
        # test set에 대한 result 저장
        np.savetxt(
            os.path.join(subresult_path) + os.path.sep + statistics + '_test' + '.txt', result_test,
            delimiter='\t', fmt=['%d', '%d', '%d', '%1.10f'], header="COMCSNO\tY\tY_hat\tp_score", comments='')
        # TRN에 대한 predict result 저장
        np.savetxt(
            os.path.join(subresult_path) + os.path.sep + statistics + '_pred' + '.txt', result_pred,
            delimiter='\t', fmt=['%d', '%d', '%1.10f'], header="COMCSNO\tY_hat\tp_score", comments='')
        # importance list 저장
        if self.verbose >= 2:
            np.savetxt(
                os.path.join(subresult_path) + os.path.sep + statistics + '_importance_group.txt',
                self.group_importance_list, delimiter='\t', fmt=['%s', '%1.10f'])
            if self.verbose == 2:
                np.savetxt(
                    os.path.join(subresult_path) + os.path.sep + statistics + '_importance_indiv.txt',
                    self.importance_list, delimiter='\t', fmt=['%s', '%1.10f'])

    def see_distribution(self, Y_test, p_test):
        # 구매주기와 관련된 feature들(84~87)에 대해서 이탈/안이탈 과 prediction 값을 비교

        if Y_test.shape[0] < 10000:
            point_size = 2
        elif Y_test.shape[0] < 100000:
            point_size = 1
        else:
            point_size = 0.1

        plt.title("brief summary")
        plt.plot([-1, 6], [self.selected_threshold, self.selected_threshold], c='gray',
                 label='threshold: %1.2f' % self.selected_threshold, linewidth=2.0)
        condition_tn = np.nonzero((Y_test==0) * (p_test < self.selected_threshold))[0]
        condition_fn = np.nonzero((Y_test==0) * (p_test > self.selected_threshold))[0]
        condition_fp = np.nonzero((Y_test==1) * (p_test < self.selected_threshold))[0]
        condition_tp = np.nonzero((Y_test==1) * (p_test > self.selected_threshold))[0]

        random.seed(1)
        plt.scatter(1 + np.random.randn(len(condition_tn)) * 0.5, p_test[condition_tn], c='blue', s=point_size)
        plt.scatter(1 + np.random.randn(len(condition_fn)) * 0.5, p_test[condition_fn], c='skyblue', s=point_size)
        plt.scatter(4 + np.random.randn(len(condition_fp)) * 0.5, p_test[condition_fp], c='salmon', s=point_size)
        plt.scatter(4 + np.random.randn(len(condition_tp)) * 0.5, p_test[condition_tp], c='magenta', s=point_size)

        plt.text(1, 0.1, '%d' % self.tn, verticalalignment='center', color='black', fontsize=20)
        plt.text(1, 0.9, '%d' % self.fp, verticalalignment='center', color='black', fontsize=20)
        plt.text(4, 0.1, '%d' % self.fn, verticalalignment='center', color='black', fontsize=20)
        plt.text(4, 0.9, '%d' % self.tp, verticalalignment='center', color='black', fontsize=20)

        plt.title('accuracy: %1.4f, power: %1.4f' % (self.precision, self.recall))
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.xlim([-1, 6])
        plt.ylim([0, 1])
        plt.ylabel('probability')
        plt.xlabel('classes')
        plt.xticks([1, 4], ('Not churn', 'churn'))
        plt.show()

    def see_density(self, Y_test, p_test):
        condition_churn = np.nonzero((Y_test == 1))[0]
        condition_notchurn = np.nonzero((Y_test == 0))[0]
        from scipy.stats import gaussian_kde
        density_c = gaussian_kde(p_test[condition_churn].reshape(-1), bw_method=.3)
        density_n = gaussian_kde(p_test[condition_notchurn].reshape(-1), bw_method=.3)
        dx = np.linspace(0., 1., len(p_test)/1000)
        plt.plot(dx, density_c(dx)/7)
        plt.fill(np.insert(dx, [0, -1], [0, 1]), np.insert(density_c(dx)/7, [0, -1], 0), color='blue', alpha=0.1)
        plt.plot(dx, density_n(dx)/3)
        plt.fill(np.insert(dx, [0, -1], [0, 1]), np.insert(density_n(dx)/3, [0, -1], 0), color='red', alpha=0.1)
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylabel('density')
        plt.xlabel('score')
        plt.yticks([])
        plt.show()

    def see_density_pred(self, p_new):
        condition_churn = np.nonzero((p_new > self.selected_threshold))[0]
        condition_notchurn = np.nonzero((p_new < self.selected_threshold))[0]
        from scipy.stats import gaussian_kde
        density_c = gaussian_kde(p_new[condition_churn].reshape(-1), bw_method=.3)
        density_n = gaussian_kde(p_new[condition_notchurn].reshape(-1), bw_method=.3)
        dx = np.linspace(0., 1., len(p_new)/100)
        plt.plot(dx, density_c(dx))
        plt.fill(np.insert(dx, [0, -1], [0, 1]), np.insert(density_c(dx), [0, -1], 0), color='blue', alpha=0.1)
        plt.plot(dx, density_n(dx))
        plt.fill(np.insert(dx, [0, -1], [0, 1]), np.insert(density_n(dx), [0, -1], 0), color='red', alpha=0.1)
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylabel('density')
        plt.xlabel('score')
        plt.yticks([])
        plt.show()
