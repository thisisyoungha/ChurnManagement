# test
import os

file_path = "D:\churn"
# input_file = "churn_learning_2_10000.txt"
# new_data_file = "churn_targeting_2_10000.txt"
input_file = "churn_learning.txt"
new_data_file = "churn_targeting.txt"

"""
----------------------------------  가이드  ------------------------------------------
file_path, input_file, new_data_file을 입력하는 곳은 맨 위에 위치(ctrl+home)
/data 에 필요한 데이터를 넣고 main_before 함수를 실행시키면 results에 결과치가 차곡차곡 저장
(단, save=True로 실행시켜야 저장됨)

results에 txt 형태의 데이터만 있다면 eval_performance 함수로 performance를 확인 가능
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


# memory를 초기화시키는 마법의 주문
def magic():
    import sys
    sys.modules[__name__].__dict__.clear()


if __name__ == '__main__':
    # main_before 함수로 전체 실행
    from utils.pipe import main

    accuracy, precision, recall = main(input_file, new_data_file,
                                       data_path=data_path, check_path=check_path, result_path=result_path,
                                       epoch=4, batch_size=100000, verbose=1, save=True, load_weights=False,
                                       class_n=1, x_offset=1, random_state=0)

    # print('acc: %1.5f, precision(정확도): %1.5f, recall(power): %1.5f' % (accuracy, precision, recall))

    """
    -----------------------  parameters  -------------------------------
    save = True: /results 에 결과물 저장됨
    ------------------------ 필요한 파일(/data) ------------------------------
    'churn_learning.txt'    :   LRN 데이터
    'churn_targeting.txt'   :   TRG 데이터
    'feature_info.csv'      :   importance 출력시 사용(verbose>=1), feature들의 정보 포함(Header 있고, COMCSNO 포함)
    'feature_info_label.csv':   importance 출력시 사용(verbose>=1), feature group에 관한 정보 포함(Header 있고, COMCSNO 없음)

    ------------------------ 저장되는 파일(/results/file name) ------------------------------
    xxx = acc, roc

    /results/file name/threshold_info.txt:      acc, roc based threshold들 값 저장

    /results/file name/xxx_pred.txt:    TRN 에 대한 예측 데이터
                                        COMCSNO, Y_hat, predicted score

    /results/file name/xxx_test.txt:    LRN_test 에 대한 데이터
                                        COMCSNO, Y, Y_hat, predicted score

    /results/file name/xxx_importance_indiv.txt:    LRN_test 에 대한 importance
                                                    individual feature name, score

    /results/file name/xxx_importance_group.txt:    LRN_test 에 대한 importance
                                                    grouped feature name, score
    """

    ####################################################################################################################
    from utils.performance import eval_performance

    # eval_performance: 결과물 출력해줌
    eval_performance(os.path.join(result_path, os.path.splitext(input_file)[0]),
                     save=True, load=False, view_plot=True, make_prob=False)
    # eval_performance(os.path.join(result_path, os.path.splitext(input_file)[0]),
    #                  save=True, load=False, base='roc', view_plot=True, make_prob=True)
    """
    -------------------------  parameters  ------------------------------
    base='roc'(or 'acc'): 추천 cutoff line의 base 선택('roc'의 power가 더 좋음)

    view_plot: plot 띄울 것인지 결정

    make_prob: 확률처럼 바꿀 것인지 결정

    save, load: save할지 load할지 여부

    ------------------------ 필요한 파일(/results/file name) ------------------------------
    /results/file name/xxx_test.txt        :   main으로 생성된 LRN의 test set에 대한 결과물
    /results/file name/xxx_pred.txt        :   main으로 생성된 TRG에 대한 결과물

    ------------------------ 저장되는 파일(/results/file name) ------------------------------
    /results/file name/xxx_fig_score.png:   summary된 내용 그래프에 함께 plot해서 저장, 
                                            score형태

    /results/file name/xxx_fig_prob.png:    확률형태

    /results/file name/xxx_results_score.txt: score형태의 결과 출력을 위한 데이터(load용)

    /results/file name/xxx_results_prob.txt: 확률형태의 결과 출력을 위한 데이터(load용)   
    """

    ####################################################################################################################
    # examine: rotation 숫자만큼 학습시켜서 통계치에 대한 평균치 구함
    # from utils.performance import examine
    #
    # input_file = os.path.join(data_path, input_file)
    # examine(input_file=input_file, data_path=data_path, epoch=10, batch_size=10000, rotation=20)

