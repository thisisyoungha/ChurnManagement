from time import time
import os
weight_file = 'churn_check.hdf5'


def main(input_file_name, new_file_name='',
            data_path='./data', check_path='./checkpoint', result_path='./results',
            class_n=1, x_offset=1, verbose=1, random_state=0,
            epoch=10, batch_size=100, save=False, load_weights=False):

    input_file = os.path.join(data_path, input_file_name)
    name, _ = os.path.splitext(input_file)
    if not new_file_name == '':
        new_data_file = os.path.join(data_path, new_file_name)

    # 전처리
    from utils.datapreprocess import Preprocess
    pre = Preprocess(random_state)  # 전처리 풀세트 패키지
    pre.class_n = class_n
    pre.data_path = data_path
    pre.x_offset = x_offset
    pre.verbose = verbose

    X_train, X_test, Y_train, Y_test= pre.preprocess(input_file)

    input_dim = X_train.shape[1] - x_offset  # network에 들어갈 input 개수 = 넣을 feature 개수
    final_output_dim = class_n  # 분류할 class 개수(여기서는 이탈할지 안할지니까 1개)

    # 이탈 모형 전체 풀 세트 패키지
    from utils.model import Churn_model
    churn = Churn_model(input_dim=input_dim, final_output_dim=final_output_dim)
    churn.check_path = check_path
    churn.model_verbose = 0
    # weights 불러오기
    if load_weights:
        from utils.datapreprocess import scale_fit
        churn.sc = scale_fit(X_train[:, x_offset:])
        churn.model.load_weights(os.path.join(check_path, weight_file))
        # churn.epoch = epoch
        # churn.batch_size = batch_size
        # churn.verbose = verbose
        # churn.model_verbose = 1
        # churn.train(X_train, X_test, Y_train, Y_test, x_offset)
    # weights 안 불러오면 새롭게 학습
    else:
        churn.epoch = epoch
        churn.batch_size = batch_size
        churn.verbose = verbose
        churn.model_verbose = 1
        churn.ES = True
        churn.train(X_train, X_test, Y_train, Y_test, x_offset)

    p_train = churn.predict(X_train, x_offset=x_offset)
    del X_train
    p_test = churn.predict(X_test, x_offset=x_offset)
    if not new_file_name == '':
        New_data = pre.perP_new(new_data_file=new_data_file)  # New_data load
        p_new = churn.predict(New_data, x_offset=x_offset)
    del churn, pre

    # 결과 추출을 위한 Extractor
    from utils.extractor import Extractor
    extractor = Extractor()
    extractor.save = save
    extractor.result_path = result_path
    extractor.data_path = data_path
    extractor.verbose = verbose
    extractor.x_offset = x_offset

    # thresholding
    extractor.thresholding(Y_train, p_train)
    del p_train

    # roc로 계산
    extractor.selected_threshold = extractor.roc_threshold

    # 각 통계치 계산
    extractor.fit(Y_test, p_test)

    extractor.print_score(Y_test, p_test)
    # extractor.roc_view(Y_test, p_test)


    extractor.selected_threshold = extractor.acc_threshold
    extractor.fit(Y_test, p_test)
    extractor.print_score(Y_test, p_test)

    import numpy as np
    print('*.5')
    extractor.selected_threshold = extractor.roc_threshold + np.std(p_test)/10*.5
    extractor.fit(Y_test, p_test)
    extractor.print_score(Y_test, p_test)

    print('*1')
    extractor.selected_threshold = extractor.roc_threshold + np.std(p_test)/10*1
    extractor.fit(Y_test, p_test)
    extractor.print_score(Y_test, p_test)

    print('*2')
    extractor.selected_threshold = extractor.roc_threshold + np.std(p_test) / 10 * 2
    extractor.fit(Y_test, p_test)
    extractor.print_score(Y_test, p_test)
    extractor.see_distribution(Y_test, p_test)


    print('*1.5')
    extractor.selected_threshold = extractor.roc_threshold + np.std(p_test) / 10 * 1.5
    extractor.fit(Y_test, p_test)
    extractor.print_score(Y_test, p_test)
    # extractor.see_distribution(Y_test, p_test)
    extractor.see_density(Y_test, p_test, p_new)

    if save:
        extractor.save_output_new(name=name, X_test=X_test, Y_test=Y_test, p_test=p_test, New_data=New_data, p_new=p_new)

    from keras import backend as K
    K.clear_session()
    return extractor.accuracy, extractor.precision, extractor.recall


def main_bf(input_file_name, new_file_name='', delete_index_g=[],
            data_path='./data', check_path='./checkpoint', result_path='./results',
            class_n=1, x_offset=1, verbose=1, random_state=0,
            epoch=10, batch_size=100, save=False, num_view=10, load_weights=False):

    input_file = os.path.join(data_path, input_file_name)
    name, _ = os.path.splitext(input_file)
    if not new_file_name == '':
        new_data_file = os.path.join(data_path, new_file_name)

    if delete_index_g:
        delete_feature = True
    else:
        delete_feature = False

    # 전처리
    from utils.datapreprocess import Preprocess
    pre = Preprocess(random_state)  # 전처리 풀세트 패키지
    pre.class_n = class_n
    pre.data_path = data_path
    pre.x_offset = x_offset
    pre.verbose = verbose
    pre.del_index_g = delete_index_g
    pre.del_feature = delete_feature

    if delete_feature:
        X_train, X_test, Y_train, Y_test, d_list = pre.preprocess(input_file)
    else:
        X_train, X_test, Y_train, Y_test = pre.preprocess(input_file)
        d_list = None

    input_dim = X_train.shape[1] - x_offset  # network에 들어갈 input 개수 = 넣을 feature 개수
    final_output_dim = class_n  # 분류할 class 개수(여기서는 이탈할지 안할지니까 1개)

    # 이탈 모형 전체 풀 세트 패키지
    from utils.model import Churn_model
    churn = Churn_model(input_dim=input_dim, final_output_dim=final_output_dim)
    churn.check_path = check_path
    churn.model_verbose = 0
    # weights 불러오기
    if load_weights:
        from utils.datapreprocess import scale_fit
        churn.sc = scale_fit(X_train[:, x_offset:])
        churn.model.load_weights(os.path.join(check_path, weight_file))
        # churn.epoch = epoch
        # churn.batch_size = batch_size
        # churn.verbose = verbose
        # churn.model_verbose = 1
        # churn.train(X_train, X_test, Y_train, Y_test, x_offset)
    # weights 안 불러오면 새롭게 학습
    else:
        churn.epoch = epoch
        churn.batch_size = batch_size
        churn.verbose = verbose
        churn.model_verbose = 1
        churn.train(X_train, X_test, Y_train, Y_test, x_offset)

    p_train = churn.predict(X_train, x_offset=x_offset)
    del X_train
    p_test = churn.predict(X_test, x_offset=x_offset)
    if not new_file_name == '':
        New_data = pre.perP_new(new_data_file=new_data_file)  # New_data load
        p_new = churn.predict(New_data, x_offset=x_offset)
    del churn, pre

    # 결과 추출을 위한 Extractor
    from utils.extractor import Extractor
    extractor = Extractor()
    extractor.save = save
    extractor.result_path = result_path
    extractor.data_path = data_path
    extractor.verbose = verbose
    extractor.x_offset = x_offset
    extractor.num_view = num_view
    extractor.feature_index = 84

    # feature delete가 가능한 버전
    extractor.delete_feature = delete_feature
    extractor.delete_index = d_list
    extractor.delete_index_g = delete_index_g

    # thresholding
    extractor.thresholding(Y_train, p_train)
    del p_train

    # roc로 계산
    extractor.selected_threshold = extractor.roc_threshold

    # 각 통계치 계산
    extractor.fit(Y_test, p_test)

    extractor.print_score(Y_test, p_test)

    extractor.importance_list, extractor.group_importance_list = extractor.importance(
        X_test[:100000, extractor.x_offset:], ((p_test[:100000, :] > extractor.selected_threshold) * 1).ravel())
    bottom_values = extractor.group_importance_list['importance'].values < 0.000001
    bottom = extractor.group_importance_list[bottom_values].sort_values(by='importance', ascending=True)

    # importance 계산
    if verbose >= 2:
        extractor.roc_view(Y_test, p_test)
        extractor.final_sale(X_test, Y_test, p_test)
        extractor.importance_view(extractor.importance_list, 'grouped')
        extractor.importance_view(extractor.importance_list, 'individual')
    if save:
        extractor.save_output(name=name, X_test=X_test, Y_test=Y_test, p_test=p_test, New_data=New_data, p_new=p_new)

    from keras import backend as K
    K.clear_session()
    return extractor.accuracy, extractor.precision, extractor.recall, bottom


def test(input_file_name, new_file_name='', delete_index_g=[],
         data_path='./data', check_path='./checkpoint', result_path='./results', group_info='feature_info_label',
         class_n=1, x_offset=1, verbose=1, random_state=0,
         epoch=10, batch_size=100, save=False, num_view=10, load_weights=False):
    input_file = os.path.join(data_path, input_file_name)
    name, _ = os.path.splitext(input_file)
    if not new_file_name == '':
        new_data_file = os.path.join(data_path, new_file_name)

    if delete_index_g:
        delete_feature = True
    else:
        delete_feature = False

    # 전처리
    from utils.datapreprocess import Preprocess
    pre = Preprocess(random_state)  # 전처리 풀세트 패키지
    pre.class_n = class_n
    pre.data_path = data_path
    pre.x_offset = x_offset
    pre.verbose = verbose
    pre.del_index_g = delete_index_g
    pre.del_feature = delete_feature
    pre.group_info = group_info

    if delete_feature:
        X_train, X_test, Y_train, Y_test, d_list = pre.preprocess(input_file)
    else:
        X_train, X_test, Y_train, Y_test = pre.preprocess(input_file)
        d_list = None
    input_dim = X_train.shape[1] - x_offset  # network에 들어갈 input 개수 = 넣을 feature 개수
    final_output_dim = class_n  # 분류할 class 개수(여기서는 이탈할지 안할지니까 1개)

    # 이탈 모형 전체 풀 세트 패키지
    from utils.model import Churn_model
    churn = Churn_model(input_dim=input_dim, final_output_dim=final_output_dim)
    churn.check_path = check_path
    churn.model_verbose = 0
    # weights 불러오기
    if load_weights:
        from utils.datapreprocess import scale_fit
        churn.sc = scale_fit(X_train)
        churn.model.load_weights(os.path.join(check_path, weight_file))
    # weights 안 불러오면 새롭게 학습
    else:
        churn.epoch = epoch
        churn.batch_size = batch_size
        churn.train(X_train, X_test, Y_train, Y_test, x_offset)
        churn.model_verbose = 0

    p_train = churn.predict(X_train, x_offset=x_offset)
    del X_train
    p_test = churn.predict(X_test, x_offset=x_offset)
    if not new_file_name == '':
        New_data = pre.perP_new(new_data_file=new_data_file)  # New_data load
        p_new = churn.predict(New_data, x_offset=x_offset)
    del churn, pre

    # 결과 추출을 위한 Extractor
    from utils.extractor import Extractor
    extractor = Extractor()
    extractor.save = save
    extractor.result_path = result_path
    extractor.data_path = data_path
    extractor.verbose = verbose
    extractor.x_offset = x_offset
    extractor.num_view = num_view
    extractor.feature_index = 84

    # feature delete가 가능한 버전
    extractor.delete_feature = delete_feature
    extractor.delete_index = d_list
    extractor.delete_index_g = delete_index_g

    # thresholding
    extractor.thresholding(Y_train, p_train)
    del p_train

    # roc로 계산
    extractor.selected_threshold = extractor.roc_threshold

    # 각 통계치 계산
    extractor.fit(Y_test, p_test)

    # importance 계산
    if verbose >= 1:
        _, group_importance_list = extractor.see_all(X_test=X_test, Y_test=Y_test, p_test=p_test)
    else:
        _, group_importance_list = extractor.importance(
            X_test[:100000, x_offset:], ((p_test[:100000, :] > extractor.selected_threshold) * 1).ravel())
    bottom_values = group_importance_list['importance'].values < 0.000001
    bottom = group_importance_list[bottom_values].sort_values(by='importance', ascending=True)

    if save:
        extractor.save_output(name=name, X_test=X_test, Y_test=Y_test, p_test=p_test, New_data=New_data, p_new=p_new)

    from keras import backend as K
    K.clear_session()
    return extractor.accuracy, extractor.precision, extractor.recall, bottom