from main import main
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool  # ここが違う
import time
import matplotlib
# matplotlib.use('Agg')

def run_test(args, additional_test=None):
    dataset, target_model, auto_encoder, alpha, selecet_feature_ratio = args
    
    auto_encoder_epochs = 100 if dataset == 'MNIST' else 1000
    auto_encoder_latent_dim = 10 if dataset == 'MNIST' else 6
    one_hot_encoding = True if dataset == 'MNIST' or dataset == 'wine' or dataset == 'gmsc' else False
    
    # MNISTの場合はLIMEで計算しない
    if dataset == 'MNIST' and auto_encoder == 'LIME': 
        return
    
    # MNISTの場合はuslimeで計算しない
    if dataset == 'MNIST' and auto_encoder == 'uslime':
        return
    
    # MNISTの場合はslimeで計算しない
    if dataset == 'MNIST' and auto_encoder == 'slime':
        return
    
    # LRの場合は多クラス問題では実行しない
    if target_model == 'LR' and (dataset == 'wine' or dataset == 'MNIST'):
        return
    
    # 安定性の追加実験を行う場合は，test_rangeを変更
    test_range = range(20) if additional_test else range(10)
    
    main(dataset=dataset,
         dataset_class_num={'breastcancer':2,
                             'hepa': 2,
                             'liver':2,
                             'adult': 2,
                             'wine': 6,
                             'credit':2,
                             'boston':'numerous',
                             'creditonehot':2,
                             'adultonehot':2,
                             'MNIST':10,
                             'compas':2,
                             'gaussian':2,
                             'german':2,
                             'gmsc':4,
                             'heart':2,
                             'heloc':2,
                             'pima':2},
         target_model=target_model,
         target_model_training=False, 
         test_range=test_range, 
         num_samples=5000,
         auto_encoder=auto_encoder,
         auto_encoder_weighting=True,
         auto_encoder_sampling=False if auto_encoder=='AE' else True,
         auto_encoder_training=False, 
         auto_encoder_epochs=auto_encoder_epochs,
         auto_encoder_latent_dim=auto_encoder_latent_dim,
         one_hot_encoding=one_hot_encoding,
         feature_selection='auto' if dataset == 'MNIST' else 'lasso_path', # https://vscode.dev/github/YasuiDaisuke/CVAE-LIME_for_Journal/blob/main/limes/test_lime/lime_base.py#L159-L169
         model_regressor=None,
         noise_std=1,
         kernel_width=None,
         label_filter=False,
         select_percent=10,
         preprocess='Minimax',
         var_threshold=0.5,
         add_condition="",
         distance_mertics='None',
         condition_from_target="",
         additional_test=additional_test,
         alpha=alpha,
         selecet_feature_ratio=selecet_feature_ratio
         )

if __name__ == "__main__":

    """
    マスター実験設定
    datasets = ['breastcancer','creditonehot','adultonehot','liver','wine','MNIST',
                'compas', 'gaussian', 'german', 'gmsc', 'heart', 'heloc', 'pima']
    target_models = ['NN', 'RF', 'SVM', 'LR']
    auto_encoders = ['CVAE', 'VAE','AE', 'uslime', 'slime', 'LIME']
    """
    import sys
    # MODEL = int(sys.argv[1])
    # DATA = int(sys.argv[2])
    # target_models_list = [['NN', 'RF', 'SVM'][MODEL]]
    # datasets_list = [['breastcancer','creditonehot','adultonehot','liver','wine'][DATA]]
    target_models_list = ['NN']
    datasets_list = ['breastcancer']
    
    auto_encoders = ['CVAE', 'VAE','AE', 'uslime', 'slime', 'LIME']
    # auto_encoders = ['CVAE', 'slime', 'LIME']


    # alpha_list = [0.2, 0.4, 0.6, 0.8, 1]
    # selecet_feature_ratio_list = [1]

    alpha_list = [1]
    selecet_feature_ratio_list = [0.8, 0.9, 1, 1.1, 1.2]
    
    MAX_RETRIES = 100  # 最大リトライ回数
    additional_test = False # 追加のテストを行うかどうか(追加テストでは，インスタンス数は10個)
    
    for dataset in datasets_list:
        for target_model in target_models_list:
            datasets = [dataset]
            target_models = [target_model]

            # Create list of arguments for each combination
            args_list = [(dataset, target_model, auto_encoder, alpha, select_feature_ratio)
                        for dataset in datasets
                        for target_model in target_models
                        for auto_encoder in auto_encoders
                        for alpha in alpha_list
                        for select_feature_ratio in selecet_feature_ratio_list]

            # 並列実行するかどうかを設定（FalseにするとargsListの最初の要素のみ実行）
            pararel = True
            
            if pararel:
                def run_with_retries(args, additional_test=additional_test):
                    """エラーが発生した場合にリトライする関数"""
                    dataset, target_model, auto_encoder, alpha, select_feature_ratio = args
                    retries = 0

                    while retries < MAX_RETRIES:
                        try:
                            print(f"Running: {dataset}, {target_model}, {auto_encoder} (Attempt {retries + 1})")
                            run_test(args, additional_test=additional_test)  # 実際のテスト関数
                            return  # 成功したら終了
                        except Exception as e:
                            print(f"Error in: {dataset}, {target_model}, {auto_encoder} -> {e}")
                            retries += 1
                            time.sleep(2)  # 少し待機してリトライ

                    print(f"Failed after {MAX_RETRIES} attempts: {dataset}, {target_model}, {auto_encoder}")


                # Run tests in parallel
                with Pool() as pool:
                    pool.map(run_with_retries, args_list)

            else:
                run_test(args_list[0])
