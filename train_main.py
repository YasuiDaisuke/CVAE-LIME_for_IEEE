from main import main
import sys

for i, dataset in enumerate(['breastcancer','creditonehot','adultonehot','liver','wine','MNIST']): 
    #['breastcancer','creditonehot','adultonehot','liver','wine','MNIST','compas', 'gaussian', 'german', 'gmsc', 'heart', 'heloc', 'pima']
    for j, target_model in enumerate(['NN', 'RF', 'SVM']): #['NN', 'RF', 'SVM', 'LR'] 
        for k, auto_encoder in enumerate(['CVAE', 'VAE', 'AE']): #['CVAE', 'VAE', 'AE', 'LIME']
            
            auto_encoder_epochs = 100 if dataset == 'MNIST' else 1000
            auto_encoder_latent_dim = 10 if dataset == 'MNIST' else 6
            auto_encoder_training = True if j == 0 else False  
            target_model_training = True  if k == 0 else False 
            one_hot_encoding = True if dataset == 'wine' or dataset == 'MNIST' or dataset == 'gmsc' else False
            
            # 全く訓練しない場合はcontinue
            if auto_encoder_training == False and target_model_training == False: 
                continue 
            
            # MNISTの場合はLIMEで計算しない
            if (dataset == 'MNIST' and auto_encoder == 'LIME') or (dataset == 'MNIST' and auto_encoder == 'AE'): 
                continue
            
            # LRの場合は多クラス問題では実行しない
            if target_model == 'LR' and (dataset == 'wine' or dataset == 'MNIST'):
                continue
            
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
                target_model_training=target_model_training,
                test_range=[0],
                num_samples=100,
                auto_encoder=auto_encoder,
                auto_encoder_weighting=True,
                auto_encoder_sampling=False if auto_encoder=='AE' else True,
                auto_encoder_training=auto_encoder_training,
                auto_encoder_epochs=auto_encoder_epochs,
                auto_encoder_latent_dim=auto_encoder_latent_dim,
                one_hot_encoding=one_hot_encoding,
                feature_selection='auto',
                model_regressor=None,
                noise_std=1,
                kernel_width=None,
                label_filter=False,
                select_percent=100,
                preprocess='Minimax',
                var_threshold=0.5,
                add_condition="",
                distance_mertics='None',
                condition_from_target="")