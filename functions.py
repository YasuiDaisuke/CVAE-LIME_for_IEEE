import numpy as np
import pickle


def target_model_loder(dataset = None,
                        target_model = None,
                        target_model_training = None,
                        X_train = None,
                        y_train = None,
                        dataset_class_num = None):
    
    # 指標を計算する関数
    def calculate_and_display_metrics(model, X_valid, y_valid, num_classes, dataset):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from sklearn.preprocessing import label_binarize

        num_class = num_classes[dataset]

        probabilities = model.predict(X_valid) #NNの時は２次元配列の確率だが，なぜか23行目がFalse

        if num_class == 2:  # 2-class classification
            y_pred = (probabilities > 0.5).astype(int)
        else:  # Multi-class classification
            print(probabilities.shape)
            if probabilities.ndim == 2:
                y_pred = np.argmax(probabilities, axis=1)
                y_valid = np.argmax(y_valid, axis=1)
            else:
                y_pred = probabilities
                y_valid = y_valid

        acc = accuracy_score(y_valid, y_pred)
        precision = precision_score(y_valid, y_pred, average='macro')
        recall = recall_score(y_valid, y_pred, average='macro')
        f1 = f1_score(y_valid, y_pred, average='macro')

        # if num_class == 2:
        #     auc = roc_auc_score(y_valid, probabilities)
        #     print(f"ROC AUC: {auc:.4f}")
        # else:  # For multi-class, compute the AUC for each class
        #     y_valid_bin = label_binarize(y_valid, classes=[i for i in range(num_class)])
        #     y_pred = label_binarize(probabilities, classes=[i for i in range(num_class)])
        #     # auc = roc_auc_score(y_valid_bin, y_pred, multi_class='ovr', average='macro')
        #     # print(f"Macro ROC AUC: {auc:.4f}")

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # モデルのファイルパス
    model_filepath = f'save_data/target_model/{target_model}_{dataset}.pkl'
    
    # X_train,y_trainから検証用を抽出
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # ターゲットモデル毎の設定(predict_probaで返す様に設定せよ)
    if target_model == 'RF':
        
        if target_model_training == True:
            if dataset_class_num[dataset] == 'numerous':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                model = model.predict
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                model = model.predict_proba
                
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
            
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
                
        return model
    
    if target_model == 'NN':
        from keras.models import Sequential, load_model
        from keras.layers import Dense

        if target_model_training == True:
            model = Sequential()

            if dataset_class_num[dataset] == 2:
                model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=30, batch_size=10, verbose=0)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                # 過学習対策
                # from keras.callbacks import EarlyStopping
                # early_stop = EarlyStopping(monitor='val_loss', patience=5)
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                with open(model_filepath, 'wb') as file:
                    pickle.dump(model, file)     
            elif dataset_class_num[dataset] == 'numerous':
                model = Sequential([
                    Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(8, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.1, verbose=0)
                # 過学習対策
                # from keras.callbacks import EarlyStopping
                # early_stop = EarlyStopping(monitor='val_loss', patience=5)
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                with open(model_filepath, 'wb') as file:
                    pickle.dump(model, file)     
            else:
                from keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=dataset_class_num[dataset])
                y_valid = to_categorical(y_valid, num_classes=dataset_class_num[dataset])
                model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(dataset_class_num[dataset], activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                # 過学習対策
                # from keras.callbacks import EarlyStopping
                # early_stop = EarlyStopping(monitor='val_loss', patience=5)
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                with open(model_filepath, 'wb') as file:
                    pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)

                
        # 出力形式の整形
        if dataset_class_num[dataset] == 'numerous':
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return probabilities
            
        elif dataset_class_num[dataset] > 2:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                return probabilities
        else:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return np.hstack((1-probabilities, probabilities))
                    
        return custom_predict_fn
    
    if target_model == 'LR':
        from sklearn.linear_model import LogisticRegression

        if target_model_training == True:
            model = LogisticRegression()

            if dataset_class_num[dataset] == 2:
                model.fit(X_train, y_train)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                with open(model_filepath, 'wb') as file:
                    pickle.dump(model, file)
            else:
                model.fit(X_train, y_train)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                with open(model_filepath, 'wb') as file:
                    pickle.dump(model, file)
                    
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)

        def custom_predict_fn(data):
            # scikit-learnのpredict_probaメソッドを使って確率を取得
            probabilities = model.predict_proba(data)
            return probabilities  # 正例の確率を返す

        return custom_predict_fn

    
    if target_model == 'CNN':
        from keras.models import Sequential, load_model
        from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
        from keras.utils import to_categorical
        
        # モデルを訓練するためのデータをCNNに合わせて変形
        X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
        X_valid = X_valid.values.reshape(X_valid.shape[0], 28, 28, 1)

        if target_model_training == True:
            model = Sequential()

            # CNNモデルの定義
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            y_train = to_categorical(y_train, 10)
            y_valid = to_categorical(y_valid, 10)

            model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=200, verbose=1)
            calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
            # モデルの保存
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
        
                # 出力形式の整形
        if dataset_class_num[dataset] == 'numerous':
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return probabilities
            
        elif dataset_class_num[dataset] > 2:
            def custom_predict_fn(data):
                import numpy as np
                # データの形状を変更
                data = data.reshape(data.shape[0], 28, 28, 1)
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                return probabilities
        else:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return np.hstack((1-probabilities, probabilities))
                    
        return custom_predict_fn, model       
        
    
    if target_model == 'DNN':
        from keras.models import Sequential, load_model
        from keras.layers import Dense

        if target_model_training == True:
            model = Sequential()
            
            if dataset_class_num[dataset] == 2:
                model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dense(10, activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(4, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
                # 過学習対策
                from keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                
            elif dataset_class_num[dataset] == 'numerous':
                model = Sequential([
                    Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(10, activation='relu'),
                    Dense(8, activation='relu'),
                    Dense(4, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                # model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
                # 過学習対策
                from keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))

            else:
                from keras.utils import to_categorical
                y_train = to_categorical(y_train, num_classes=dataset_class_num[dataset])
                y_valid = to_categorical(y_valid, num_classes=dataset_class_num[dataset])
                model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dense(10, activation='relu'))
                model.add(Dense(8, activation='relu'))
                model.add(Dense(4, activation='relu'))
                model.add(Dense(dataset_class_num[dataset], activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                # model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
                # 過学習対策
                from keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5)
                model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_data=(X_valid, y_valid))
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
        
        # 出力形式の整形
        if dataset_class_num[dataset] == 'numerous':
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return probabilities
            
        elif dataset_class_num[dataset] > 2:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                return probabilities
        else:
            def custom_predict_fn(data):
                import numpy as np
                # Kerasモデルのpredictメソッドを使って、2D配列の確率を取得
                probabilities = model.predict(data, verbose=0)
                # 確率を1D配列に変換
                return np.hstack((1-probabilities, probabilities))
                
        return custom_predict_fn
    
    if target_model == 'SVM':

        if target_model_training == True:
            if dataset_class_num[dataset] == 'numerous':
                from sklearn.svm import SVR
                model = SVR()
                model.fit(X_train, y_train)
                model = model.predict
            else:
                from sklearn.svm import SVC
                model = SVC(probability=True, random_state=42)
                model.fit(X_train, y_train)
                calculate_and_display_metrics(model, X_valid, y_valid, dataset_class_num, dataset)
                model = model.predict_proba
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
                
        return model
    
    if target_model == 'GBM':
        
        if target_model_training == True:
            if dataset_class_num[dataset] == 'numerous':
                from lightgbm import LGBMRegressor, early_stopping
                model = LGBMRegressor(max_depth=4, colsample_bytree=0.5, 
                        reg_lambda=0.5, reg_alpha=0.5, 
                        importance_type="gain", random_state=100)
                # model.fit(X_train, y_train)
                # 過学習対策
                model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_valid, y_valid)])
            else:
                from lightgbm import LGBMClassifier, early_stopping
                model = LGBMClassifier(max_depth=4, colsample_bytree=0.5, 
                        reg_lambda=0.5, reg_alpha=0.5, 
                        importance_type="gain", random_state=100)
                # model.fit(X_train, y_train)
                # 過学習対策
                model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_valid, y_valid)])
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
        
        if dataset_class_num[dataset] == 'numerous':
            def predict_fn(X):
                if len(X.shape)==1:
                    return model.predict(X.reshape(1,-1))[0]
                else:
                    return model.predict(X)
        else:
            # 出力形式の整形
            def predict_fn(X):
                if len(X.shape)==1:
                    return model.predict_proba(X.reshape(1,-1))[0]
                else:
                    return model.predict_proba(X)

        return predict_fn
    
    if target_model == 'XGB':
        import xgboost as xgb

        if target_model_training == True:
            if dataset_class_num[dataset] == 'numerous':
                model = xgb.XGBRegressor(random_state=42, eval_metric="rmse")
                # model.fit(X_train, y_train)
                # 過学習対策
                model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)])
                model = model.predict
            else:
                model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
                # model.fit(X_train, y_train)
                # 過学習対策
                model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)])
                model = model.predict_proba
            with open(model_filepath, 'wb') as file:
                pickle.dump(model, file)
        
        elif target_model_training == False:
            with open(model_filepath, 'rb') as file:
                model = pickle.load(file)
                
        return model

def create_folder(path):
    """指定したパスにフォルダを作成します。"""
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        
def quantize_matrix(mat):
    '''
    ・入力は0~1で正規化している必要がある
    ・最大値と最小値を4分割し,0~3の整数値に置き換える
    '''
    # 入力の形状を保持しておきます
    original_shape = mat.shape

    # 行列を1次元に変換
    flattened = mat.ravel()

    # 各要素に対して量子化を適用
    import numpy as np
    quantized = np.piecewise(flattened, 
                             [flattened < 0.25, 
                              (0.25 <= flattened) & (flattened < 0.5),
                              (0.5 <= flattened) & (flattened < 0.75),
                              0.75 <= flattened],
                             [0, 1, 2, 3])

    # 元の形状に戻して返却
    return quantized.reshape(original_shape)

def get_highest_probability_index(data):
    '''
    クラス分類の出力値のargmaxを返す
    '''
    import numpy as np
    # Numpy配列に変換
    arr = np.array(data)

    # 最大値のインデックスを返す
    return np.argmax(arr)

def jaccard_index(set_a, set_b):
    """ジャカード指数を計算する関数"""
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

def calculate_jaccard_for_all_combinations(features_list):
    """複数の特徴のリストから、すべての組み合わせのジャカード指数を計算する関数"""
    n = len(features_list)
    jaccard_values = []

    for i in range(n):
        for j in range(i+1, n):
            set_a = set(features_list[i])
            set_b = set(features_list[j])
            jaccard_values.append(jaccard_index(set_a, set_b))

    return jaccard_values

def extract_top_n_percent(samples, weights, labels, n):
    import numpy as np
    # 上位n%のインデックス数を計算
    top_n_idx_count = int(len(weights) * n / 100)

    # weightsの上位n%のインデックスを取得
    top_indices = np.argsort(weights)[-top_n_idx_count:]

    # 上位n%に対応するsamplesとlabelsを取得
    top_samples = samples[top_indices]
    top_labels = labels[top_indices]
    top_weights = weights[top_indices]

    return top_samples, top_weights, top_labels

# データをCSVファイルに書き込む関数
def append_to_csv(filename, data, column_names):
    '''
    filename:出力先のパス
    data:リスト形式で渡す
    column_names:リスト形式でカラム名を渡す
    '''
    import csv
    # ファイルが存在しない場合、ヘッダを書き込む
    try:
        with open(filename, 'r') as f:
            pass
    except FileNotFoundError:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(["dataset", "target_model", "auto_encoder_weighting", "auto_encoder_sampling", "auto_encoder", "instance_no", "noise_std", "kernel_width"])
            writer.writerow(column_names)

    # データを追加
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def filtering(neighborhood_data, neighborhood_labels, weights, label):
    # neighborhood_labels の label の列の値が各行の最大値である行のインデックスを抽出
    import numpy as np
    max_indices = np.argmax(neighborhood_labels, axis=1)
    selected_rows = np.where(max_indices == label)[0]

    # 対応する行の weights と neighborhood_data を再定義
    weights = weights[selected_rows]
    neighborhood_data = neighborhood_data[selected_rows]
    neighborhood_labels = neighborhood_labels[selected_rows]
    
    return neighborhood_data, neighborhood_labels, weights


def small_sample_reduce(X, y, reduce_percent):
    import numpy as np
    # 値が0のインデックスを取得
    zero_indices = np.where(y == 0)[0]

    # 削除する行数を計算
    delete_count = int(len(zero_indices) * reduce_percent)  # 例として50%を削除

    # 削除するインデックスをランダムに選択
    delete_indices = np.random.choice(zero_indices, delete_count, replace=False)

    # Xとyから指定されたインデックスを削除
    X_new = X.drop(delete_indices)
    y_new = y.drop(delete_indices)
    
    return X_new, y_new

def iAUC(model, surrogate_model, test_data):
    import numpy as np
    from sklearn.metrics import log_loss
    
    def masking_function(x, s, mask_value=-999):
        """
        Return x after replacing features where s_i = 0 by mask_value.
        """
        return [xi if si else mask_value for xi, si in zip(x, s)]

    def topn_attributions(lime_explanation, n):
        """
        Return a binary mask where the top n% of features have a value of 1.
        """
        n_features = len(lime_explanation)
        top_n = int(np.ceil(n * n_features / 100))
        sorted_indices = sorted(range(n_features), key=lambda i: abs(lime_explanation[i]), reverse=True)
        s = [0] * n_features
        for idx in sorted_indices[:top_n]:
            s[idx] = 1
        return s

    def compute_iAUC(model, surrogate_model, test_data):
        '''
        lime_explanation:特徴量と係数が入ったリスト
        
        '''
        iAUC_values = []
        
        for n in range(101):  # From 0 to 100
            log_likelihoods = []
            
            for x, y, exp in test_data:
                # Get LIME explanation
                # exp = explainer.explain_instance(x, model.predict_proba).as_list()
                lime_explanation = [value for feature, value in exp.local_exp]
                
                # Mask top n% of features
                s = topn_attributions(lime_explanation, n)
                x_masked = masking_function(x, s)
                
                # Compute the log-likelihood using surrogate model
                probs = surrogate_model.predict_proba([x_masked])
                log_likelihood = -log_loss([y], probs, labels=class_names)
                log_likelihoods.append(log_likelihood)
                
            expected_log_likelihood = np.mean(log_likelihoods)
            iAUC_values.append(expected_log_likelihood)
        
        return np.trapz(iAUC_values)  # Compute the area under the curve
    
    return compute_iAUC(model, surrogate_model, test_data)

# オブジェクトをファイルに書き出す関数
def write_object_to_file(obj, filename):
    import dill
    with open(filename, 'wb') as f:
        dill.dump(obj, f)

# ファイルからオブジェクトを読み出す関数
def read_object_from_file(filename):
    import dill
    with open(filename, 'rb') as f:
        return dill.load(f)
    
# MNISTを含むexpの画像を表示するプログラム
def MNIST_exp(test_instance, exp, auto_encoder, target_model, instance_no, label, X_train, auto_encoder_latent_dim, original_model=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

            
    # L1,L2ノルムを計算
    def calculate_gini(data):
        values = [abs(value) for value in data]  # 値の絶対値を取得
        n = len(values)
        
        # Gini係数を計算
        values_sorted = sorted(values)
        cumulative_values = [0] + values_sorted
        sum_values = sum(values_sorted)
        
        gini_numerator = sum((i + 1) * values_sorted[i] for i in range(n))
        gini_denominator = n * sum_values
        
        gini_coefficient = (2 * gini_numerator) / gini_denominator - (n + 1) / n
        
        return gini_coefficient
    
    # サンプルのデータ (特徴量の番号, 特徴量のデータ)
    features_list = list(exp.values())[0]

    # 1. 特徴量の番号でソート
    sorted_features = sorted(features_list, key=lambda x: x[0])

    # 2. 特徴量のデータだけ取り出し
    data_only = [item[1] for item in sorted_features]

    
    # 3. このデータを28x28の形状にreshape
    image_data1 = np.array(data_only).reshape(28, 28)

    # Test instanceのデータを28x28の形状にreshape
    image_data2 = np.array(test_instance).reshape(28, 28)

    # 最初の画像を描画して保存
    plt.imshow(image_data1, cmap='gray')
    plt.axis('off')
    plt.savefig(f"temp/exp_{auto_encoder}_{auto_encoder_latent_dim}_{target_model}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # グラフをクリア

    # 2番目の画像を描画して保存
    plt.imshow(image_data2, cmap='gray')
    plt.axis('off')
    plt.savefig(f"temp/testinstance_{auto_encoder}_{auto_encoder_latent_dim}_{target_model}.png", bbox_inches='tight', pad_inches=0)
    plt.close()  # グラフをクリア
    
    if original_model != None:
        from tf_keras_vis.gradcam import Gradcam
        from tf_keras_vis.saliency import Saliency
        from tf_keras_vis.utils import normalize

        test_instance = np.array(test_instance).reshape(1, 28, 28, 1)

        def model_modifier(m):
            return m[:, label]
        
        # Generate GradCam AttributionMAP
        gradcam = Gradcam(original_model)
        cam = gradcam(model_modifier, seed_input=test_instance, penultimate_layer=-1)  # You may need to specify a different layer
        cam = normalize(cam)
        
        # score関数を作成します。この関数は、モデルの出力から特定のクラスのスコアを取得します。
        def get_score(output):
            return output[:, label]
        
        # Generate IntegralGradient AttributionMAP
        integral_gradient = Saliency(original_model)
        mask = integral_gradient(score=get_score, seed_input=test_instance)
        mask = normalize(mask)
        
        # Compute SHAP values
        import shap
        from lime import lime_image
        from lime.wrappers.scikit_image import SegmentationAlgorithm
        import tensorflow as tf
        tf.config.experimental_run_functions_eagerly(False)
        background_data_indices = np.random.choice(len(X_train), 100, replace=False)
        background_data = X_train[background_data_indices]
        background_data = np.array(background_data).reshape(len(background_data), 28, 28, 1)
        explainer = shap.DeepExplainer(original_model, background_data)
        shap_values = explainer.shap_values(test_instance)
        shap_image = shap_values[0][0].reshape(28, 28)
        
        # LIMEの説明を生成
        from lime import lime_image
        from lime.wrappers.scikit_image import SegmentationAlgorithm
        import tensorflow as tf
        tf.config.experimental_run_functions_eagerly(True)

        explainer = lime_image.LimeImageExplainer()
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
        # LIMEはモデルの出力として確率値を必要とするため、predict_proba関数を作成します。
        def predict_proba(images):
            # グレースケール画像に変換
            images = np.mean(images, axis=-1, keepdims=True)
            preds = original_model.predict(images)
            return preds
        explanation = explainer.explain_instance(test_instance.reshape(28, 28), predict_proba, top_labels=5, hide_color=0, num_samples=1000, segmentation_fn=segmenter)
        temp, _ = explanation.get_image_and_mask(label, positive_only=False, hide_rest=False, num_features=10, min_weight=0.01)
        
        # Plot and save all images
        fig, axs = plt.subplots(1, 6, figsize=(20,5))

        axs[0].imshow(image_data2, cmap='gray')
        axs[0].axis('off')
        
        axs[1].imshow(image_data1, cmap='jet')
        axs[1].axis('off')
        
        axs[2].imshow(temp, cmap='jet') # LIMEの説明
        axs[2].axis('off')
        
        axs[3].imshow(shap_image, cmap='jet') # Displaying SHAP
        axs[3].axis('off')
        
        axs[4].imshow(cam[0], cmap='jet')
        axs[4].axis('off')
        
        axs[5].imshow(mask[0], cmap='jet')
        axs[5].axis('off')
        

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(f"save_data/MNIST_png/combined_GradCAM_and_IG_{auto_encoder}_{auto_encoder_latent_dim}_{target_model}_{label}_{instance_no}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        import math
        return  sum(abs(x) for x in data_only), math.sqrt(sum(x**2 for x in data_only)), calculate_gini(data_only)

    else:
        # 2つの画像を読み込む
        img1 = Image.open(f"temp/exp_{auto_encoder}_{auto_encoder_latent_dim}_{target_model}.png")
        img2 = Image.open(f"temp/testinstance_{auto_encoder}_{auto_encoder_latent_dim}_{target_model}.png")

        # 画像を横に結合
        combined_img = Image.new('RGB', (img1.width + img2.width, img1.height))
        combined_img.paste(img2, (0, 0))
        combined_img.paste(img1, (img1.width, 0))

        # 結合した画像を保存
        combined_img.save(f"save_data/MNIST_png/combined_{auto_encoder}_{auto_encoder_latent_dim}_{target_model}_{label}_{instance_no}.png")
        
        import math
        return  sum(abs(x) for x in data_only), math.sqrt(sum(x**2 for x in data_only)), calculate_gini(data_only)


def tabel_exp_visualize(exp, exp_setting, total_feature_name):
    import pandas as pd
    import os
    
    # CSVファイルが存在するか確認し、存在しない場合は新しいファイルを作成します。
    if not os.path.exists("save_data/visualize_table_exp/" + exp_setting + '.csv'):
        columns = total_feature_name
        df = pd.DataFrame(columns=columns)
        df.to_csv(f"save_data/visualize_table_exp/{exp_setting}.csv", index=False)

    # 各クラスの説明をデータフレームに追加します。
    class_exp = list(exp.local_exp.values())[0]
    data = {}
    for feature_name in total_feature_name:
        data[feature_name] = 0
            
    for feature, weight in class_exp:
        feature_name = total_feature_name[feature]
        data[feature_name] = weight

    df = pd.DataFrame([data])
    
    # 結果をCSVファイルに追記します。
    df.to_csv(f"save_data/visualize_table_exp/{exp_setting}.csv", mode='a', header=False, index=False)
    
    # L1,L2ノルムを計算
    def calculate_norms_and_gini(data):
        l1_norm = 0
        l2_norm = 0
        
        values = [abs(value) for value in data.values()]  # 値の絶対値を取得
        n = len(values)
        
        # ノルムを計算
        for value in values:
            l1_norm += abs(value)  # L1ノルムの計算
            l2_norm += value**2    # L2ノルムの計算（平方和）
        
        l2_norm = l2_norm**0.5  # L2ノルムの平方根を計算
        
        # Gini係数を計算
        values_sorted = sorted(values)
        cumulative_values = [0] + values_sorted
        sum_values = sum(values_sorted)
        
        gini_numerator = sum((i + 1) * values_sorted[i] for i in range(n))
        gini_denominator = n * sum_values
        
        gini_coefficient = (2 * gini_numerator) / gini_denominator - (n + 1) / n
        
        # ベクトルのヒストグラムを計算
        hist, bin_edges = np.histogram(values, bins=100, density=True)
        
        # 確率分布を計算
        probabilities = hist * np.diff(bin_edges)
        
        # 確率が0の部分を取り除く
        probabilities = probabilities[probabilities > 0]
        
        # エントロピーを計算
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # 変動係数(Coefficient of Variation)を計算
        CV = np.std(values) / np.mean(values)
        
        return l1_norm, l2_norm, gini_coefficient, entropy, CV
    
    return calculate_norms_and_gini(data)

def count_below_threshold(VAR, VAR_threshold):
    """
    Count the number of elements in VAR that are below a certain threshold.
    
    Parameters:
    - VAR: numpy array or list, the input data
    - VAR_threshold: float, the threshold value
    
    Returns:
    - int, number of elements below the threshold
    """
    import numpy as np
    # VARがリストの場合、NumPy配列に変換
    if isinstance(VAR, list):
        VAR = np.array(VAR)
    
    # 閾値以下の要素のブールインデックスを取得
    below_threshold_indices = VAR < VAR_threshold
    
    # 閾値以下の要素数をカウント
    count = np.sum(below_threshold_indices)
    
    return count
    
def simple_WSL(X, y, weights):
    import statsmodels.api as sm
    import numpy as np
    
    # 定数項 (intercept) を追加
    X = sm.add_constant(X)

    # WLSモデルの作成とフィット
    model = sm.WLS(y, X, weights=weights).fit()

    # p値が0.05以下の係数の数をカウント
    significant_coeffs = (model.pvalues < 0.05).sum()
    
    # 結果のサマリーを表示
    # print(model.summary())
    # print(significant_coeffs)
    
    return significant_coeffs


def RSS_TSS_fn(easy_model, neighborhood_data, labels_column, weights):
    import numpy as np
    # モデルを使って予測値を計算
    predictions = easy_model.predict(neighborhood_data)
    
    # 実際の値と予測値の差（残差）を計算
    residuals = labels_column - predictions
    
    # 残差を重みで重み付けする（weightsが指定されている場合）
    weighted_residuals = residuals**2 * weights
    
    WRSR = np.sum(weighted_residuals) / np.sum(residuals**2)
    
    WRSR2 = WRSR / np.mean(weights)
    
    # 重みとWRSRの相関係数を確認する
    import scipy.stats
    Corr = {}
    Corr['Peason'] = np.corrcoef(residuals**2, weights)[0, 1]
    Corr['spearman'], spearman_p_value = scipy.stats.spearmanr(residuals**2, weights)
    Corr['Kendor'], kendall_p_value = scipy.stats.kendalltau(residuals**2, weights)
    
    # 残差の二乗和（RSS）を計算
    RSS = np.sum(weighted_residuals)
    
    # 実際の値の平均を計算
    average_label = np.average(labels_column, weights=weights)
    
    # 実際の値と平均との差（全体の変動）を計算
    total_var = (labels_column - average_label) ** 2
    
    # 重みのヒストグラム
    from functions import Histgram
    Histgram(weights)
    
    
    # 全平方和（TSS）を計算
    TSS = np.sum(total_var * weights)
    
    # R2の計算
    R2 = 1 - RSS / TSS
    
    return RSS, TSS, R2, WRSR, WRSR2, Corr


def weight_RSS_fn(easy_model, neighborhood_data, labels_column, weights, auto_encoder):
    import numpy as np
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import scipy.stats as stats
    # モデルを使って予測値を計算
    predictions = easy_model.predict(neighborhood_data)
    
    # 実際の値と予測値の差（残差）を計算
    residuals = np.abs(labels_column - predictions)

    # 散布図を作成
    plt.figure()
    plt.hexbin(residuals, weights, gridsize=50, cmap='Blues')
    cb = plt.colorbar(label='Sample Num')
    plt.xlabel('Residuals')
    plt.ylabel('Weights')
    plt.title('Hexbin Plot of Residuals vs Weights')
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 1)
    plt.close()

def Histgram(data):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.hist(data, bins=30, alpha=0.7, label='A')
    plt.title('Weight Frequency (Adult Dataset:LIME)')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

    # 統計量の計算
    mean_value = np.mean(data)
    median_value = np.median(data)
    std_deviation = np.std(data)
    min_value = np.amin(data)
    max_value = np.amax(data)
    total = np.sum(data)
    count = data.size

    # 統計量のテキストを設定
    stats_text = f"Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\nStd: {std_deviation:.2f}\nMin: {min_value:.2f}\nMax: {max_value:.2f}\nTotal: {total:.2f}\nCount: {count:.2f}"

    # 統計量のテキストをグラフに追加
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top')

    plt.xlim(0, 1)
    plt.ylim(0, 1000)

    # plt.savefig('save_data/histgram.png')
    plt.close()
    
# もとの特徴量の並び順に整形
def get_exp_list_all(exp, explainer):
    import numpy as np
    exp_list_all = np.zeros(len(explainer.feature_names))
    for col_id, val in exp.local_exp[exp.top_labels[0]]:
        exp_list_all[col_id] = val
    return exp_list_all

# 訓練データでPCAのモデルを構築し，生成サンプルの分布を可視化する
def PCA_plot(training_data, exp_setting, test_instance, PCA_training=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import os
    import re
    
    exp_setting_noid = re.sub(r'_[^_]*$', '', exp_setting)
    Path2gendata = 'save_data/test_samples/' + exp_setting + '/gen_data.csv'
    Path2PCA = 'save_data/PCA_models/' + exp_setting_noid + '.pkl'

    # データの読み込み
    gendata = np.loadtxt(Path2gendata, delimiter=',')

    # PCAのモデルの構築またはロード
    if PCA_training:
        pca = PCA(n_components=2)
        pca.fit(training_data)
        # PCAモデルを保存
        if not os.path.exists('save_data/PCA_models/'):
            os.makedirs('save_data/PCA_models/')
        with open(Path2PCA, 'wb') as f:
            pickle.dump(pca, f)
    else:
        with open(Path2PCA, 'rb') as f:
            pca = pickle.load(f)

    # PCAによる次元削減
    training_data_pca = pca.transform(training_data)
    gendata_pca = pca.transform(gendata)
    test_instance_pca = pca.transform(test_instance.reshape(1, -1))
    dataset = exp_setting.split('_')[4]
    explainer = exp_setting.split('_')[5]
    id = exp_setting.split('_')[8]
    
    # 説明対象インスタンスと生成サンプルの重心の差の大きさの計算
    gendata_dist = np.linalg.norm(np.mean(gendata_pca - test_instance_pca, axis=0))
    
    # グラフの描画
    plt.figure(figsize=(8, 6))
    # plt.scatter(training_data_pca[:, 0], training_data_pca[:, 1], color='blue', label='Training Data')
    # plt.scatter(gendata_pca[:100, 0], gendata_pca[:100, 1], color='green', label='Generated Data')
    # plt.scatter(test_instance_pca[:, 0], test_instance_pca[:, 1], color='red', label='Test Instance')
    plt.scatter(training_data_pca[:, 0], training_data_pca[:, 1], color='blue', marker='.', s=10, label='Training Data')
    plt.scatter(gendata_pca[:100, 0], gendata_pca[:100, 1], color='green', marker='x', label='Generated Data')
    plt.scatter(test_instance_pca[:, 0], test_instance_pca[:, 1], color='red', marker='o', s=100, label='Test Instance')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA Plot(Data:{dataset}, Explainer:{explainer}), Ave. dist.:{gendata_dist:.2f}')
    # plt.text(3, 0.5, f'{gendata_dist:.2f}')
    plt.legend()
    plt.savefig(f'save_data/visualize_gensamples/PCA_plot(Data:{dataset},Explainer:{explainer},No:{id}).png', dpi=500)
    # plt.show()

    gendata_dist = 0
    
    return gendata_dist

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def jaccard_distance(usecase):
    sim = []
    for l in usecase:
        i_sim = []
        for j in usecase:
            i_sim.append(1 - jaccard_similarity(l, j))
        sim.append(i_sim)
    return sim

def calc_jaccard(seed_results_list, num_features):

    # 前処理
    results_list = []
    for seed_results in seed_results_list:
        repeat_exp_list = [v for k, v in seed_results.items()][0] # 辞書の値のリストを取得
        top_k_features = sorted(repeat_exp_list, key=lambda x: abs(x[1]), reverse=True)[:num_features] # 絶対値の大きい上位k個の特徴量を取得
        top_k_features_list = [feat[0] for feat in top_k_features]
        results_list.append(top_k_features_list)

    # Jaccard Distance の計算
    jaccard_matrix = jaccard_distance(results_list)

    # 計算内容の後処理
    jaccard_matrix = np.array(jaccard_matrix)
    # 対角成分を除外して平均を取る（スカラー値に変換）
    # 上三角行列のインデックス取得（対角成分を除く）
    # 上三角部分（対角成分を除く）のインデックスを取得
    triu_indices = np.triu_indices(len(jaccard_matrix), k=1)
    # 上三角部分の要素を取得
    upper_triangle_values = jaccard_matrix[triu_indices]  # ここでエラーが起きないように修正
    # Jaccard Distance の平均値と中央値を計算
    mean_jaccard = np.mean(upper_triangle_values)
    median_jaccard = np.median(upper_triangle_values)

    return mean_jaccard, median_jaccard


def standard_deviation(coefs_list, num_features):

    compare_std_list = []
    compare_mean_list = []

    for k, coefs in enumerate(coefs_list):
        # print(len(coefs))
        # print(coefs)
        compare_mean = []
        compare_std = []
        for i in range(num_features):
            # print([print(len(Mx)) for Mx in coefs])
            # f1 = np.array([Mx[i] for Mx in coefs])
            f1 = np.array([coefs_list[j][i] for j in range(len(coefs_list))])  # 修正
            # print(f'mean for f{i}:', np.mean(f1))
            # print(f'std for f{i}:', np.std(f1))
            compare_mean.append(np.mean(f1))
            compare_std.append(np.std(f1))


        compare_std_list.append([np.mean(np.array(compare_std))])
        compare_mean_list.append([np.mean(np.array(compare_mean))])

    compare_std_t = np.mean(np.array(compare_std), axis=0)
    compare_mean_t =  np.mean(np.array(compare_mean), axis=0)

    return compare_std_t, compare_mean_t

def calc_standard_deviation(seed_results_list, num_features):

    # 前処理
    # Step 0: seed_results_list の辞書から特徴量のリストを取得
    results_list = []
    for seed_results in seed_results_list:
        results_list.append([v for k, v in seed_results.items()][0]) # 辞書の値のリストを取得
    
    # Step 1: すべての試行で出現した特徴量インデックスを取得（昇順で統一）
    all_features = sorted(set(f[0] for trial in results_list for f in trial))

    # Step 2: 各試行の特徴量を統一された順番に並べる
    coefs_list = []
    for trial in results_list:
        feature_dict = dict(trial)  # (特徴量インデックス, 係数) を辞書に変換
        sorted_coefs = [feature_dict.get(f, 0.0) for f in all_features]  # 存在しない特徴量は0
        coefs_list.append(sorted_coefs)

    # Step 3: 標準偏差の計算
    compare_std_t, compare_mean_t = standard_deviation(coefs_list, num_features)

    return compare_std_t, compare_mean_t



def noisy_instance_maker(instance, noise_strength):
    noise = np.random.normal(0, noise_strength, len(instance))
    noisy_instance = instance + noise
    return noisy_instance


"""
SHAPとIGとの安定性の比較を実施
"""