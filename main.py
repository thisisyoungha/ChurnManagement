#git test

import os

file_path = "D:\churn"
# input_file = "churn_learning_2_10000.txt"
# new_data_file = "churn_targeting_2_10000.txt"
input_file = "churn_learning.txt"
new_data_file = "churn_targeting.txt"

"""
----------------------------------  ê°€?´ë“œ  ------------------------------------------
file_path, input_file, new_data_file???…ë ¥?˜ëŠ” ê³³ì? ë§??„ì— ?„ì¹˜(ctrl+home)
/data ???„ìš”???°ì´?°ë? ?£ê³  main_before ?¨ìˆ˜ë¥??¤í–‰?œí‚¤ë©?results??ê²°ê³¼ì¹˜ê? ì°¨ê³¡ì°¨ê³¡ ?€??
(?? save=Trueë¡??¤í–‰?œì¼œ???€?¥ë¨)

results??txt ?•íƒœ???°ì´?°ë§Œ ?ˆë‹¤ë©?eval_performance ?¨ìˆ˜ë¡?performanceë¥??•ì¸ ê°€??
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


# memoryë¥?ì´ˆê¸°?”ì‹œ?¤ëŠ” ë§ˆë²•??ì£¼ë¬¸
def magic():
    import sys
    sys.modules[__name__].__dict__.clear()


if __name__ == '__main__':
    # main_before ?¨ìˆ˜ë¡??„ì²´ ?¤í–‰
    from utils.pipe import main

    accuracy, precision, recall = main(input_file, new_data_file,
                                       data_path=data_path, check_path=check_path, result_path=result_path,
                                       epoch=4, batch_size=100000, verbose=1, save=True, load_weights=False,
                                       class_n=1, x_offset=1, random_state=0)

    # print('acc: %1.5f, precision(?•í™•??: %1.5f, recall(power): %1.5f' % (accuracy, precision, recall))

    """
    -----------------------  parameters  -------------------------------
    save = True: /results ??ê²°ê³¼ë¬??€?¥ë¨
    ------------------------ ?„ìš”???Œì¼(/data) ------------------------------
    'churn_learning.txt'    :   LRN ?°ì´??
    'churn_targeting.txt'   :   TRG ?°ì´??
    'feature_info.csv'      :   importance ì¶œë ¥???¬ìš©(verbose>=1), feature?¤ì˜ ?•ë³´ ?¬í•¨(Header ?ˆê³ , COMCSNO ?¬í•¨)
    'feature_info_label.csv':   importance ì¶œë ¥???¬ìš©(verbose>=1), feature group??ê´€???•ë³´ ?¬í•¨(Header ?ˆê³ , COMCSNO ?†ìŒ)

    ------------------------ ?€?¥ë˜???Œì¼(/results/file name) ------------------------------
    xxx = acc, roc

    /results/file name/threshold_info.txt:      acc, roc based threshold??ê°??€??

    /results/file name/xxx_pred.txt:    TRN ???€???ˆì¸¡ ?°ì´??
                                        COMCSNO, Y_hat, predicted score

    /results/file name/xxx_test.txt:    LRN_test ???€???°ì´??
                                        COMCSNO, Y, Y_hat, predicted score

    /results/file name/xxx_importance_indiv.txt:    LRN_test ???€??importance
                                                    individual feature name, score

    /results/file name/xxx_importance_group.txt:    LRN_test ???€??importance
                                                    grouped feature name, score
    """

    ####################################################################################################################
    from utils.performance import eval_performance

    # eval_performance: ê²°ê³¼ë¬?ì¶œë ¥?´ì¤Œ
    eval_performance(os.path.join(result_path, os.path.splitext(input_file)[0]),
                     save=True, load=False, view_plot=True, make_prob=False)
    # eval_performance(os.path.join(result_path, os.path.splitext(input_file)[0]),
    #                  save=True, load=False, base='roc', view_plot=True, make_prob=True)
    """
    -------------------------  parameters  ------------------------------
    base='roc'(or 'acc'): ì¶”ì²œ cutoff line??base ? íƒ('roc'??powerê°€ ??ì¢‹ìŒ)

    view_plot: plot ?„ìš¸ ê²ƒì¸ì§€ ê²°ì •

    make_prob: ?•ë¥ ì²˜ëŸ¼ ë°”ê? ê²ƒì¸ì§€ ê²°ì •

    save, load: save? ì? load? ì? ?¬ë?

    ------------------------ ?„ìš”???Œì¼(/results/file name) ------------------------------
    /results/file name/xxx_test.txt        :   main?¼ë¡œ ?ì„±??LRN??test set???€??ê²°ê³¼ë¬?
    /results/file name/xxx_pred.txt        :   main?¼ë¡œ ?ì„±??TRG???€??ê²°ê³¼ë¬?

    ------------------------ ?€?¥ë˜???Œì¼(/results/file name) ------------------------------
    /results/file name/xxx_fig_score.png:   summary???´ìš© ê·¸ë˜?„ì— ?¨ê»˜ plot?´ì„œ ?€?? 
                                            score?•íƒœ

    /results/file name/xxx_fig_prob.png:    ?•ë¥ ?•íƒœ

    /results/file name/xxx_results_score.txt: score?•íƒœ??ê²°ê³¼ ì¶œë ¥???„í•œ ?°ì´??load??

    /results/file name/xxx_results_prob.txt: ?•ë¥ ?•íƒœ??ê²°ê³¼ ì¶œë ¥???„í•œ ?°ì´??load??   
    """

    ####################################################################################################################
    # examine: rotation ?«ìë§Œí¼ ?™ìŠµ?œì¼œ???µê³„ì¹˜ì— ?€???‰ê· ì¹?êµ¬í•¨
    # from utils.performance import examine
    #
    # input_file = os.path.join(data_path, input_file)
    # examine(input_file=input_file, data_path=data_path, epoch=10, batch_size=10000, rotation=20)

