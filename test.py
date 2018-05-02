import os
file_path = "D:\churn"
input_file = "churn_learning_2_10000.txt"
new_data_file = "churn_targeting_2_10000.txt"

"""
----------------------------------  가이드  ------------------------------------------
test용 파일
group별로 backward elimination
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


if __name__ == '__main__':
    """
                            "test"
    네트워크 안에서 각 feature의 영향력을 알아보기 위해서 
    group 통째로 feature를 삭제시켜 학습시킬 수 있게끔 만든 함수
    
    -------------------------  parameters  ------------------------------
    delete_index_g  :   group별 label에서 삭제할 index 표시(/data/feature_info_label.csv 참조)
    random_state    :   같은 수를 넣으면 똑같은 train, test split을 얻음
    ---------------------------- output ------------------------------
    precision, recall   :   특정 feature 그룹을 제거시키고 얻은 precision과 recall
    bottom  :   이번 학습에서 importance가 0.000001보다 작은 feature 그룹(다음 loop에서 제거될 친구들)
    """
    from utils.pipe import test
    import pandas as pd
    import numpy as np
    from utils.datapreprocess import load_info
    import gc
    from sklearn.preprocessing import OneHotEncoder
    group_list = load_info(data_path, 'feature_info_label')
    group_names = group_list['group'].values[1:]

    ohe = OneHotEncoder()
    ohe.fit(np.arange(0, 54, dtype=int).reshape(-1, 1))

    dict_rot = {}
    selected_features_sum = np.repeat(0, 54)
    # main_bf rotation
    for rotation in range(10):
        delete_index_g = []     # 매번 삭제할 group index
        dict_subrot = {}
        # sub rotation
        for i in range(10):
            print('\nrotation %d - %d' % (rotation+1, i+1))
            accuracy, precision, recall, bottom = test(
                input_file_name=input_file, delete_index_g=delete_index_g,
                data_path=data_path, check_path=check_path, result_path=result_path,
                epoch=10, batch_size=100000, verbose=0, save=False,
                num_view=30, class_n=1, x_offset=1, random_state=np.random.randint(i, i+50))
            index = []
            for value in bottom['feature'].values:
                index.append(np.where(group_names == value)[0][0])
            print('acc: %1.4f, precision: %1.4f, recall: %1.4f\n' % (accuracy, precision, recall) +
                  'worst importance:', bottom['feature'].values,
                  '\ntheir indices: ', index)
            delete_index_g = np.append(delete_index_g, index).astype(int).tolist()
            dict_subrot[i] = index
            gc.collect()

        temp = ohe.transform(np.array(delete_index_g).reshape(-1, 1)).toarray()
        temp_sum = np.sum(temp, axis=0).astype(int)
        selected_features_sum += temp_sum
        dict_rot['%d_rot' % rotation] = dict_subrot

    print('\n최종\n', dict_rot)
    selected_features = pd.DataFrame(data={'feature_group': group_names, 'num_of_selection': selected_features_sum})
    print(selected_features)
    selected_features.to_csv(os.path.join(result_path, 'test_feature_info.csv'), sep=',', header=True)
