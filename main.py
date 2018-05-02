#git test

import os

file_path = "D:\churn"
# input_file = "churn_learning_2_10000.txt"
# new_data_file = "churn_targeting_2_10000.txt"
input_file = "churn_learning.txt"
new_data_file = "churn_targeting.txt"

"""
----------------------------------  가?�드  ------------------------------------------
file_path, input_file, new_data_file???�력?�는 곳�? �??�에 ?�치(ctrl+home)
/data ???�요???�이?��? ?�고 main_before ?�수�??�행?�키�?results??결과치�? 차곡차곡 ?�??
(?? save=True�??�행?�켜???�?�됨)

results??txt ?�태???�이?�만 ?�다�?eval_performance ?�수�?performance�??�인 가??
--------------------------------------------------------------------------------------
"""
check_path = os.path.join(file_path, 'checkpoint')
data_path = os.path.join(file_path, 'data')
result_path = os.path.join(file_path, 'results')
if not os.path.exists(check_path):
    os.mkdir(check_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)
if not os.path.exists(data_path):
    os.mkdir(data_path)


# memory�?초기?�시?�는 마법??주문
def magic():
    import sys
    sys.modules[__name__].__dict__.clear()


if __name__ == '__main__':
    # main_before ?�수�??�체 ?�행
    from utils.pipe import main

    accuracy, precision, recall = main(input_file, new_data_file,
                                       data_path=data_path, check_path=check_path, result_path=result_path,
                                       epoch=4, batch_size=100000, verbose=1, save=True, load_weights=False,
                                       class_n=1, x_offset=1, random_state=0)

    # print('acc: %1.5f, precision(?�확??: %1.5f, recall(power): %1.5f' % (accuracy, precision, recall))

    """
    -----------------------  parameters  -------------------------------
    save = True: /results ??결과�??�?�됨
    ------------------------ ?�요???�일(/data) ------------------------------
    'churn_learning.txt'    :   LRN ?�이??
    'churn_targeting.txt'   :   TRG ?�이??
    'feature_info.csv'      :   importance 출력???�용(verbose>=1), feature?�의 ?�보 ?�함(Header ?�고, COMCSNO ?�함)
    'feature_info_label.csv':   importance 출력???�용(verbose>=1), feature group??관???�보 ?�함(Header ?�고, COMCSNO ?�음)

    ------------------------ ?�?�되???�일(/results/file name) ------------------------------
    xxx = acc, roc

    /results/file name/threshold_info.txt:      acc, roc based threshold??�??�??

    /results/file name/xxx_pred.txt:    TRN ???�???�측 ?�이??
                                        COMCSNO, Y_hat, predicted score

    /results/file name/xxx_test.txt:    LRN_test ???�???�이??
                                        COMCSNO, Y, Y_hat, predicted score

    /results/file name/xxx_importance_indiv.txt:    LRN_test ???�??importance
                                                    individual feature name, score

    /results/file name/xxx_importance_group.txt:    LRN_test ???�??importance
                                                    grouped feature name, score
    """

    ####################################################################################################################
    from utils.performance import eval_performance

    # eval_performance: 결과�?출력?�줌
    eval_performance(os.path.join(result_path, os.path.splitext(input_file)[0]),
                     save=True, load=False, view_plot=True, make_prob=False)
    # eval_performance(os.path.join(result_path, os.path.splitext(input_file)[0]),
    #                  save=True, load=False, base='roc', view_plot=True, make_prob=True)
    """
    -------------------------  parameters  ------------------------------
    base='roc'(or 'acc'): 추천 cutoff line??base ?�택('roc'??power가 ??좋음)

    view_plot: plot ?�울 것인지 결정

    make_prob: ?�률처럼 바�? 것인지 결정

    save, load: save?��? load?��? ?��?

    ------------------------ ?�요???�일(/results/file name) ------------------------------
    /results/file name/xxx_test.txt        :   main?�로 ?�성??LRN??test set???�??결과�?
    /results/file name/xxx_pred.txt        :   main?�로 ?�성??TRG???�??결과�?

    ------------------------ ?�?�되???�일(/results/file name) ------------------------------
    /results/file name/xxx_fig_score.png:   summary???�용 그래?�에 ?�께 plot?�서 ?�?? 
                                            score?�태

    /results/file name/xxx_fig_prob.png:    ?�률?�태

    /results/file name/xxx_results_score.txt: score?�태??결과 출력???�한 ?�이??load??

    /results/file name/xxx_results_prob.txt: ?�률?�태??결과 출력???�한 ?�이??load??   
    """

    ####################################################################################################################
    # examine: rotation ?�자만큼 ?�습?�켜???�계치에 ?�???�균�?구함
    # from utils.performance import examine
    #
    # input_file = os.path.join(data_path, input_file)
    # examine(input_file=input_file, data_path=data_path, epoch=10, batch_size=10000, rotation=20)

