"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path, Lasso
from sklearn.utils import check_random_state


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None,
                 auto_encoder_setting=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.auto_encoder_setting = auto_encoder_setting
        
    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None,
                                   inverse=None,
                                   data_row=None,
                                   label_filter=None,
                                   alpha=1,
                                   ):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        # weights = self.kernel_fn(distances)
        
        # X_test_predic(=CVAEの条件)に関する処理
        from functions import quantize_matrix, get_highest_probability_index
        # データが1次元の場合に2次元に変換
        if len(data_row.shape) == 1:
            lamda = data_row.reshape(1, -1)
        # X_testに対する説明対象モデルの出力を取得
        X_test_output = self.auto_encoder_setting['predict_fn'](lamda)
        
        if self.auto_encoder_setting['one_hot_encoding'] == True:
            # one-hotエンコーディングする場合
            if self.auto_encoder_setting['mode'] == 'classification':
                X_test_condition = get_highest_probability_index(X_test_output)
                # X_test_predictを確認し、スカラー値の場合は配列に変換 (AE.py l.190対策)
                if np.isscalar(X_test_condition):
                    X_test_condition = np.array([[X_test_condition]]) 
                # X_test_conditionの型チェックし,np.float32で無ければnp.float32に変換(AE.py l.281対策)
                if X_test_condition.dtype != np.float32:
                    X_test_condition = X_test_condition.astype(np.float32)
                    # X_test_conditionをone_hotエンコーディングする．
                    # 1. y_trainをone-hotエンコーディング
                from tensorflow.keras.utils import to_categorical
                num_classes = int(self.auto_encoder_setting['dataset_class_num'])  # クラス数を取得
                X_test_condition = to_categorical(X_test_condition, num_classes=num_classes)
            else:
                from tensorflow.keras.utils import to_categorical
                X_test_condition = quantize_matrix(X_test_output)
                X_test_condition = to_categorical(X_test_condition, num_classes=4)
        else:
            X_test_condition = get_highest_probability_index(X_test_output)
            
        self.auto_encoder_setting['X_test_condition'] = X_test_condition
                
            
        # オートエンコーダからのロード
        from limes.cvae_lime.auto_encoders.AE import AE_load
        neighborhood_data, weights, yss, Active_latent_dim = AE_load(X_test = data_row,
                                    inverse = inverse,
                                    auto_encoder_setting = self.auto_encoder_setting
                                    )
        
        #　ラベルによるフィルタリング
        if label_filter == True:
            from functions import filtering
            neighborhood_data, yss, weights = filtering(neighborhood_data, yss, weights, label)
        
        ## yss(生成サンプルの出力)に関する処理
        # for regression, the output should be a one-dimensional array of predictions
        if self.auto_encoder_setting['mode'] != "classification":
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]
            
        # labels_columnにyssを代入
        labels_column = yss[:, label]
                
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=alpha, fit_intercept=True,
                                    random_state=self.random_state)
            
        easy_model = model_regressor
        
        # 最終的なサンプルの抽出
        exp_setting = 'turb_'+str(self.auto_encoder_setting['auto_encoder_sampling'])+'_filter_'+str(self.auto_encoder_setting['filtering'])+'_'+str(self.auto_encoder_setting['dataset'])+'_'+str(self.auto_encoder_setting['auto_encoder'])+'_'+str(self.auto_encoder_setting['latent_dim'])+'_'+str(self.auto_encoder_setting['select_percent'])+'_'+str(self.auto_encoder_setting['instance_no'])
        
        if self.auto_encoder_setting['dataset'] != 'MNIST':
            np.savetxt('save_data/test_samples/' + exp_setting + '/gen_data.csv', neighborhood_data, delimiter=',')
            np.savetxt('save_data/test_samples/' + exp_setting + '/gen_data_label.csv', labels_column, delimiter=',')
            np.savetxt('save_data/test_samples/' + exp_setting + '/weights.csv', weights, delimiter=',')
        
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        
        # RSSの計算
        from functions import RSS_TSS_fn
        RSS, TSS, R2, WRSR, WRSR2, Corr = RSS_TSS_fn(easy_model, neighborhood_data[:, used_features], labels_column, weights)

        # weightとRSSの関係をプロット
        # from functions import weight_RSS_fn
        # weight_RSS_fn(easy_model, neighborhood_data, labels_column, weights, self.auto_encoder_setting['auto_encoder'])
        
        # 単純な重み付き線形モデルとして評価する
        # from functions import simple_WSL
        # significant_coeffs = simple_WSL(neighborhood_data[:, used_features], labels_column, weights)
        significant_coeffs = None
        
        additional_score = {'local_pred':local_pred, 
                           'Active_latent_dim':Active_latent_dim, 
                           'significant_coeffs':significant_coeffs, 
                           'RSS':RSS, 
                           'TSS':RSS, 
                           'WRSR':WRSR, 
                           'WRSR2':WRSR2, 
                           'Corr':Corr}
        
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_), key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score,
                local_pred,
                additional_score
                )
