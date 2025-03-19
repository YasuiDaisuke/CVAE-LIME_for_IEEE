import sys
sys.path.append('/home/CVAE-LIME20230802/')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # v1 APIのWARNING以上のメッセージも非表示にする
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import *
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.simplefilter(action='ignore', category=FutureWarning)

"追加実験用"
# import shap
# from captum.attr import IntegratedGradients


def main(dataset=None,
        dataset_class_num=None,
        target_model=None,
        target_model_training=None,
        test_range=None,
        num_samples=None,
        auto_encoder=None,
        auto_encoder_weighting=None,
        auto_encoder_sampling=None,
        auto_encoder_training=None,
        auto_encoder_epochs=None,
        auto_encoder_latent_dim=None,
        one_hot_encoding=None,
        feature_selection=None,
        model_regressor=None,
        noise_std=None,
        kernel_width=None,
        label_filter=None,
        select_percent=None,
        preprocess=None,
        var_threshold=None,
        add_condition=None,
        distance_mertics=None,
        condition_from_target=None,
        additional_test=False,
        selecet_feature_ratio=1,
        alpha=0.1
        ):
    
    # print(f'dataset:{dataset}_AE:{auto_encoder}_target_model:{target_model}')
    
    ## 前処理済みデータセットのロード
    # データセットをCSVファイルから読み込む
    if dataset == 'MNIST':
        from sklearn.preprocessing import StandardScaler
        mnist = tf.keras.datasets.mnist # MNISTデータをダウンロードする
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data() # データを訓練セットとテストセットに分ける
        train_images_flat = train_images.reshape(train_images.shape[0], 28*28) # 訓練データとテストデータを1次元に変換
        test_images_flat = test_images.reshape(test_images.shape[0], 28*28)
        all_images_flat = np.vstack([train_images_flat, test_images_flat]) # 画像データを結合
        all_labels = np.concatenate([train_labels, test_labels]) # ラベルデータを結合
        
        if preprocess == 'Standard':
            scaler = StandardScaler() # データの正規化
            scaler.fit(train_images_flat) # 訓練データに対してスケーラーをフィット
            all_images_flat_normalized = scaler.transform(all_images_flat) # すべての画像データを変換
            
        elif preprocess == 'Minimax':
            all_images_flat_normalized = all_images_flat / 255.0 # データの正規化
            
        # DataFrameに変換
        data = pd.DataFrame(all_images_flat_normalized)
        data['target'] = all_labels
    else:
        data = pd.read_csv(f'dataset/{dataset}.csv')
    
    # データセットを特徴量とターゲットに分割
    X = data.drop('target', axis=1)
    y = data['target']

    # データセットをトレーニングセットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ## ターゲットモデルの学習又はロード(predict_probaで返す)
    if target_model == 'CNN':
        model, original_model = target_model_loder(dataset = dataset,
                                    target_model = target_model,
                                    target_model_training = target_model_training,
                                    X_train = X_train,
                                    y_train = y_train,
                                    dataset_class_num = dataset_class_num)
    else:
        model = target_model_loder(dataset = dataset,
                                    target_model = target_model,
                                    target_model_training = target_model_training,
                                    X_train = X_train,
                                    y_train = y_train,
                                    dataset_class_num = dataset_class_num)
        original_model = None

    ## 実験結果格納用のCSVを定義
    df = pd.DataFrame(columns=['dataset',
                                'weighting_fn',
                                'epoch_num',
                                'latent_size',
                                'num_samples',
                                'select_percent',
                                'instance_no',
                                'predict_label',
                                'label', 
                                'r2', 
                                'mse',
                                'element1',
                                'element3',
                                'process_time',
                                'target_model',
                                'local_output',
                                'Active_latent_dim',
                                'L1',
                                'L2',
                                'RSS',
                                'TSS',
                                'WRSR',
                                'WRSR2',
                                'Peason',
                                'spearman',
                                'Kendor',
                                'gendata_dist',
                                'gini',
                                'entropy',
                                'CV'])
    output_path = 'save_data/test_result/turb_'+str(auto_encoder_sampling)+'_filter_'+str(label_filter)+'_'+str(dataset)+'_'+str(auto_encoder)+'_'+str(auto_encoder_latent_dim)+'_'+str(select_percent)+'_'+str(target_model)+str(add_condition)+'alpha_'+str(alpha)+'_ratio_'+str(selecet_feature_ratio)+'.csv'
    df.to_csv(output_path)
    
    # LIME explainerを作成
    if auto_encoder ==  'LIME':
        from limes.test_lime.lime_tabular import LimeTabularExplainer
        
        # セッティングを辞書に格納
        setting_dic = {
                        'auto_encoder':auto_encoder,
                        'auto_encoder_weighting':None,
                        'auto_encoder_sampling':None,
                        'auto_encoder_training':None,
                        'epochs':None,
                        'dataset':dataset,
                        'latent_dim':None,
                        'num_samples':num_samples,
                        'instance_no':None,
                        'X_test':X_test,
                        'predict_fn':model,
                        'mode':['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0],
                        'filtering':label_filter,
                        'select_percent':None,
                        'y_test':y_test,
                        'dataset_class_num':dataset_class_num[dataset],
                        }
        explainer = LimeTabularExplainer(X_train.values, 
                                        mode=['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0], 
                                        training_labels=y_train, 
                                        feature_names=X_train.columns.tolist(),
                                        # random_state=42,
                                        lime_setting=setting_dic,
                                        feature_selection=feature_selection,
                                        kernel_width=kernel_width)
    elif auto_encoder == "uslime":
        from limes.uslime.lime_tabular import USLimeTabularExplainer

        setting_dic = {
                        'auto_encoder':auto_encoder,
                        'auto_encoder_weighting':None,
                        'auto_encoder_sampling':None,
                        'auto_encoder_training':None,
                        'epochs':None,
                        'dataset':dataset,
                        'latent_dim':None,
                        'num_samples':num_samples,
                        'instance_no':None,
                        'X_test':X_test,
                        'predict_fn':model,
                        'mode':['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0],
                        'filtering':label_filter,
                        'select_percent':None,
                        'y_test':y_test,
                        'dataset_class_num':dataset_class_num[dataset],
                        }

        explainer = USLimeTabularExplainer(X_train.values, 
                                        mode=['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0], 
                                        training_labels=y_train, 
                                        feature_names=X_train.columns.tolist(),
                                        # random_state=42,
                                        lime_setting=setting_dic,
                                        feature_selection=feature_selection,
                                        kernel_width=kernel_width)
        
    elif auto_encoder == "slime":
        from limes.slime.slime.lime_tabular import LimeTabularExplainer

        setting_dic = {
                        'auto_encoder':auto_encoder,
                        'auto_encoder_weighting':None,
                        'auto_encoder_sampling':None,
                        'auto_encoder_training':None,
                        'epochs':None,
                        'dataset':dataset,
                        'latent_dim':None,
                        'num_samples':num_samples,
                        'instance_no':None,
                        'X_test':X_test,
                        'predict_fn':model,
                        'mode':['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0],
                        'filtering':label_filter,
                        'select_percent':None,
                        'y_test':y_test,
                        'dataset_class_num':dataset_class_num[dataset],
                        }

        explainer = LimeTabularExplainer(X_train.values, 
                                        mode=['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0], 
                                        training_labels=y_train, 
                                        feature_names=X_train.columns.tolist(),
                                        # random_state=42,
                                        lime_setting=setting_dic,
                                        feature_selection=feature_selection,
                                        kernel_width=kernel_width)
    # elif auto_encoder == "shap":
    #     def explain_with_shap(model, X_sample, X_background):
    #         explainer = shap.KernelExplainer(model.predict, X_background)  # 背景データを指定
    #         shap_values = explainer.shap_values(X_sample)
    #         return shap_values
    
    # elif auto_encoder == "IG":
    #     ig = IntegratedGradients(model)
    #     X_tensor = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True)
    #     attr, _ = ig.attribute(X_tensor, target=0, return_convergence_delta=True)
    #     return attr.detach().numpy()
        
    else:
        from limes.cvae_lime.lime_tabular import LimeTabularExplainer

        setting_dic = {'auto_encoder':auto_encoder,
                        'auto_encoder_weighting':auto_encoder_weighting,
                        'auto_encoder_sampling':auto_encoder_sampling,
                        'auto_encoder_training':auto_encoder_training,
                        'epochs':auto_encoder_epochs,
                        'dataset':dataset,
                        'latent_dim':auto_encoder_latent_dim,
                        'num_samples':num_samples,
                        'instance_no':None,
                        'X_test':X_test,
                        'predict_fn':model,
                        'mode':['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0],
                        'filtering':label_filter,
                        'select_percent':select_percent,
                        'y_test':y_test,
                        'dataset_class_num':dataset_class_num[dataset],
                        'one_hot_encoding':one_hot_encoding,
                        'noise_std':noise_std,
                        'kernel_width':kernel_width,
                        'VAR_threshold':var_threshold,
                        'add_condition':add_condition,
                        'condition_from_target':condition_from_target,
                        'distance_mertics':distance_mertics,
                        }
        
        explainer = LimeTabularExplainer(X_train.values, 
                                        mode=['regression' if dataset_class_num[dataset]=='numerous' else 'classification'][0], 
                                        training_labels=y_train, 
                                        feature_names=X_train.columns.tolist(),
                                        # random_state=42,
                                        auto_encoder_setting=setting_dic,
                                        feature_selection=feature_selection
                                        )
    
    # テストセットの一部のインスタンスに対して説明を取得
    for i in test_range:
        
        exp_setting = 'turb_'+str(setting_dic['auto_encoder_sampling'])+'_filter_'+str(setting_dic['filtering'])+'_'+str(setting_dic['dataset'])+'_'+str(setting_dic['auto_encoder'])+'_'+str(setting_dic['latent_dim'])+'_'+str(setting_dic['select_percent'])+'_'+str(i)
        
        if dataset != 'MNIST':
            # 生成サンプルの保存フォルダの作成
            from functions import create_folder
            create_folder('save_data/test_samples/' + exp_setting)
        

        '''
        追加実験
        '''
        if additional_test == False:
            #　計算時間の計測開始
            start = time.process_time()
            
            #　説明の生成
            if auto_encoder == 'shap': #追加実験
                pass
            elif auto_encoder == 'IG': #追加実験
                pass
            else:
                exp = explainer.explain_instance(X_test.values[i],
                                                model,
                                                num_samples=num_samples,
                                                instance_no=i,
                                                model_regressor=model_regressor,
                                                top_labels = 1,
                                                label_filter=label_filter,
                                                num_features={'breastcancer':10*selecet_feature_ratio, ##全勝
                                                            'liver':10*selecet_feature_ratio, ##RF以外全勝 
                                                            'wine':10*selecet_feature_ratio, ##NNのみ勝ち
                                                            'creditonehot':30*selecet_feature_ratio, ##全勝
                                                            'adultonehot':15*selecet_feature_ratio, #15だとNNのみ勝ち #保留
                                                            'MNIST':784,  ##全勝
                                                            # 'compas':2,
                                                            # 'gaussian':2,
                                                            # 'german':2,
                                                            # 'gmsc':4,
                                                            # 'heart':2,
                                                            # 'heloc':2,
                                                            # 'pima':2
                                                            }[dataset],
                                                alpha=alpha
                                            )
            
            # 評価値の算出
            process_time = time.process_time() - start
            noise_result_dict = {}
            
        else:
            #　説明の生成
            exp = explainer.explain_instance(X_test.values[i],
                                            model,
                                            num_samples=num_samples,
                                            instance_no=i,
                                            model_regressor=model_regressor,
                                            top_labels = 1,
                                            label_filter=label_filter,
                                            num_features={'breastcancer':10, ##全勝
                                                        'liver':10, ##RF以外全勝 
                                                        'wine':10, ##NNのみ勝ち
                                                        'creditonehot':30, ##全勝
                                                        'adultonehot':15, #15だとNNのみ勝ち #保留
                                                        'MNIST':784,  ##全勝
                                                        # 'compas':2,
                                                        # 'gaussian':2,
                                                        # 'german':2,
                                                        # 'gmsc':4,
                                                        # 'heart':2,
                                                        # 'heloc':2,
                                                        # 'pima':2
                                                        }[dataset],
                                            alpha=alpha
                                            )
            # 上記k個の特徴量で評価
            num_featur = 6

            # # 繰り返し実行安定性について評価
            #region
            # seed_results_list = []
            # for seed in range(3):
            #     # np.random.seed(seed)
            #     repeat_exp = explainer.explain_instance(
            #                                     X_test.values[i],
            #                                     model,
            #                                     num_samples=num_samples,
            #                                     instance_no=i,
            #                                     model_regressor=model_regressor,
            #                                     top_labels = 1,
            #                                     label_filter=label_filter,
            #                                     num_features=784
            #                                     )
            #     seed_results_list.append(repeat_exp.local_exp)
            # mean_jaccard, median_jaccard = calc_jaccard(seed_results_list, num_features)
            # compare_std_t, compare_mean_t = calc_standard_deviation(seed_results_list, num_features)
            # seed_results_dict = {
            #                     "mean_jaccard":mean_jaccard,
            #                     "median_jaccard":median_jaccard,
            #                     "compare_std_t":compare_std_t,
            #                     "compare_mean_t":compare_mean_t
            #                      }
            #endregion
            
            # ノイズロバスト性について評価
            noise_result_dict = {}
            
            #　計算時間の計測開始
            start = time.process_time()
            
            for noise_strength in np.arange(0, 0.4, 0.1):
                noise_result_list = [exp.local_exp]
                noisy_instance = noisy_instance_maker(X_test.values[i], noise_strength)
                noisy_exp = explainer.explain_instance(
                                                noisy_instance,
                                                model,
                                                num_samples=num_samples,
                                                instance_no=i,
                                                model_regressor=model_regressor,
                                                top_labels = 1,
                                                label_filter=label_filter,
                                                num_features={'breastcancer':10, ##全勝
                                                            'liver':10, ##RF以外全勝 
                                                            'wine':10, ##NNのみ勝ち
                                                            'creditonehot':30, ##全勝
                                                            'adultonehot':15, #15だとNNのみ勝ち #保留
                                                            'MNIST':784,  ##全勝
                                                            # 'compas':2,
                                                            # 'gaussian':2,
                                                            # 'german':2,
                                                            # 'gmsc':4,
                                                            # 'heart':2,
                                                            # 'heloc':2,
                                                            # 'pima':2
                                                            }[dataset]
                                                )
                noise_result_list.append(noisy_exp.local_exp)
                # Jaccard Distanceと標準偏差の計算
                noise_mean_jaccard, noise_median_jaccard = calc_jaccard(noise_result_list, num_featur)
                noise_compare_std_t, noise_compare_mean_t = calc_standard_deviation(noise_result_list, num_featur)
                noise_result_dict[noise_strength] = {'noise_mean_jaccard':noise_mean_jaccard, 
                                                    'noise_median_jaccard':noise_median_jaccard, 
                                                    'noise_compare_std_t':noise_compare_std_t, 
                                                    'noise_compare_mean_t':noise_compare_mean_t}
                process_time = time.process_time() - start
                

        # expの内部の整理(uslime互換性取る)
        if isinstance(exp.score, dict):
            _ = exp.score
            exp.score = [v for k, v in _.items()][0]
            _ = exp.local_pred
            exp.local_pred = [v for k, v in _.items()][0]
        # endregion

        process_time = f'{process_time:.1f}'
        local_output = model(X_test.values[i].reshape(1, -1)).reshape(-1)
        score = exp.score
        score = f'{score:.3f}'
        mse = 0.5*(exp.local_pred - np.max(local_output))**2
        mse = f'{mse[0]:.3f}'
        if auto_encoder != 'LIME':
            Active_latent_dim = exp.additional_score['Active_latent_dim']
        else:
            Active_latent_dim = ''
            exp.additional_score['Active_latent_dim'] = ''
        
        # print(f'instance:{i}, score:{score}, mse:{mse}, class:{np.argmax(local_output)}, Active_latent_dim:{Active_latent_dim}')

        if dataset != 'MNIST':
            from functions import tabel_exp_visualize
            name = 'turb_'+str(setting_dic['auto_encoder_sampling'])+'_filter_'+str(setting_dic['filtering'])+'_'+str(setting_dic['dataset'])+'_'+str(setting_dic['auto_encoder'])+'_'+str(setting_dic['latent_dim'])+'_'+str(setting_dic['select_percent'])+'_'+str(target_model)
            L1, L2, gini, entropy, CV = tabel_exp_visualize(exp,name, X_train.columns.tolist())
            #訓練データ，生成サンプル及び説明対象インスタンスのPCAによる比較を行い，生成サンプルの説明対象インスタンスとの距離を計算する
            gendata_dist = PCA_plot(X_train, exp_setting, X_test.values[i], PCA_training = True if i == 0 else False)
            gendata_dist = None
            
        else:
            # 説明の可視化とL1及びL2ノルムの獲得
            from functions import MNIST_exp
            L1, L2, gini = MNIST_exp(X_test.values[i], exp.local_exp, auto_encoder, target_model, i,y_test.values[i], X_train.values, auto_encoder_latent_dim, original_model=original_model)
            gendata_dist = ''
            entropy = ''
            CV = ''
                
        
        # 実験結果の保存
        df = pd.read_csv(output_path, index_col=0)
        df.to_csv(output_path)
        temp_data = [dataset,
                            auto_encoder,
                            int(auto_encoder_epochs),
                            int(auto_encoder_latent_dim),
                            int(num_samples),
                            select_percent,
                            int(i),
                            np.argmax(local_output) if dataset_class_num[dataset]!='numerous' else local_output[0],
                            y_test.values[i],
                            score,
                            mse,
                            auto_encoder_sampling,
                            label_filter,
                            process_time,
                            target_model,
                            min(local_output),
                            Active_latent_dim,
                            auto_encoder_latent_dim,
                            L1,
                            L2,
                            exp.additional_score['RSS'],
                            exp.additional_score['TSS'],
                            exp.additional_score['WRSR'],
                            exp.additional_score['WRSR2'],
                            exp.additional_score['Corr']['Peason'],
                            exp.additional_score['Corr']['spearman'],
                            exp.additional_score['Corr']['Kendor'],
                            gendata_dist,
                            gini,
                            entropy,
                            CV,
                            alpha, #追加実験（解釈可能性のトレードオフ）
                            selecet_feature_ratio #追加実験（解釈可能性のトレードオフ）
                            ]
        
        # temp_data+=[v for k, v in seed_results_dict.items()] #追加実験（安定性）
        for noise, result_dict in noise_result_dict.items(): #追加実験（ノイズロバスト性）
            temp_data+=[v for k, v in result_dict.items()]

        columns=['dataset', 'weighting_fn', 'epoch_num', 'latent_size', 'num_samples',
                    'select_percent','instance_no','predict_label','label', 'r2', 'mse',
                    'element1','element3','process_time','target_model','local_output',
                    'Active_latent_dim','auto_encoder_latent_dim','L1','L2','RSS','TSS',
                    'WRSR','WRSR2', 'Peason', 'spearman', 'Kendor','gendata_dist',
                    'gini','entropy','CV',
                    'alpha', #追加実験（解釈可能性のトレードオフ）
                    'selecet_feature_ratio' #追加実験（解釈可能性のトレードオフ）
                    ]
        
        # columns+=[k for k, v in seed_results_dict.items()] # 追加実験（安定性）
        for noise, result_dict in noise_result_dict.items(): # 追加実験（ノイズロバスト性）
            columns+=[f'noise{noise:.1g}:{k}' for k, v in result_dict.items()]
        
        temp = pd.DataFrame([temp_data],
                    columns=columns
                                )

        temp = temp.astype({col: 'int' for col in temp.columns if temp[col].dtype == 'bool'})
        df = pd.concat([df, temp], axis=0)
        df.to_csv(output_path)

    
if __name__ == '__main__':
    
    import sys
    DATASET = 0
    target = 0
    AE = 5
    
    dataset = ['breastcancer','creditonehot','adultonehot','liver','wine','MNIST'][DATASET]
    target_model=['NN', 'RF', 'SVM'][target]
    auto_encoder=['CVAE', 'VAE', 'AE', 'LIME', 'uslime', 'slime'][AE]
    additional_test = False
                
    auto_encoder_epochs = 100 if dataset == 'MNIST' else 1000
    auto_encoder_latent_dim = 10 if dataset == 'MNIST' else 6
    one_hot_encoding = True if dataset == 'MNIST' or dataset == 'wine' else False
    auto_encoder_sampling = True if auto_encoder == 'CVAE' or auto_encoder == 'VAE' else False
    feature_selection = 'lasso_path' if auto_encoder == 'slime' else 'auto'
    
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
        test_range=range(1),
        num_samples=5000,
        auto_encoder=auto_encoder,
        auto_encoder_weighting=True,
        auto_encoder_sampling=auto_encoder_sampling,
        auto_encoder_training=False,
        auto_encoder_epochs=auto_encoder_epochs,
        auto_encoder_latent_dim=auto_encoder_latent_dim,
        one_hot_encoding=one_hot_encoding,
        feature_selection=feature_selection,
        model_regressor=None,
        noise_std=1,
        kernel_width=None, #DAE,LIME用
        label_filter=False,
        select_percent=100,
        preprocess='Minimax', # 標準化'Standard'　or 正規化'Minimax'
        var_threshold=0.5, #活性潜在変数か否かの分散ベクトルの閾値
        add_condition="", #条件ベクトルに入力ベクトルを追加 [0,1,2,3,4,5] 
        distance_mertics='None', #距離計算方法 NoneならEuclid
        condition_from_target="", #条件ベクトルへの入力を説明対象モデルの出力クラスに変更 #""ならデータセットのラベルを学習する．
        additional_test=True,#IEEE追加実験
        alpha=0.1,
        selecet_feature_ratio=1
        )

