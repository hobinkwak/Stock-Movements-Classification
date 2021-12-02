import datetime
from itertools import combinations
import os
import pickle
import time

from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn_genetic.callbacks import TimerStopping, ConsecutiveStopping, ProgressBar
import joblib

from skopt import BayesSearchCV
from skopt import space

from genetic_selection import GeneticSelectionCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils.feature_selection import feature_select_by_vif, post_hoc_feature_select


class Ensemble:
    def __init__(self, mode='grid', fs_mode='stat', vif=3000, mi=True, pi=True):
        self.mode = mode
        self.fs_mode = fs_mode
        self.vif_thresholds = vif
        self.n_jobs = 4
        self.cv = 3
        self.mi = mi
        self.pi = pi
        if self.mode == 'genetic':
            self.rf = {'model': ExtraTreesClassifier(n_jobs=self.n_jobs),
                       'param': {
                                   'n_estimators': Integer(30, 100),
                                 'max_depth': Integer(2, 10),
                                 'min_samples_split' : Integer(2, 20),
                                 'min_samples_leaf' : Integer(2,20),
                                }
                       
                       }

            self.lg = {'model': LogisticRegression(max_iter=1000, penalty='elasticnet', solver='saga'),
                       'param': {"C": Continuous(1e-4, 1e3, distribution='log-uniform'),
                                 "l1_ratio": Continuous(0, 1)}
                      }

            self.svc = {'model': SVC(probability=True),
                        'param': {'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                                  'gamma': Continuous(1e-4, 1e4, distribution='log-uniform'),
                                  'C': Continuous(1e-4, 1e4, distribution='log-uniform'),
                                  'degree': Integer(1, 6)}
                        }

            self.xgb = {'model': XGBClassifier(n_jobs=self.n_jobs, eval_metric='logloss',
                                               use_label_encoder=False),
                        'param': {
                                    'n_estimators': Integer(30, 100),
                                  'max_depth': Integer(2, 10),
                                  'reg_alpha': Continuous(1e-3, 1, distribution='log-uniform'),
                                  'reg_lambda': Continuous(1e-3, 1, distribution='log-uniform'),
                                  'subsample' : Continuous(0.6, 1, distribution='log-uniform'),
                                  'min_child_weight': Integer(1, 5),
                                  'learning_rate' : Continuous(0.01, 0.2, distribution='log-uniform')
                                  }
                        
                        }

            self.lgbm = {'model': LGBMClassifier(n_jobs=self.n_jobs, objective='binary', max_depth=-1),
                         'param': {
                                 'n_estimators': Integer(30, 200),
                                   'subsample' : Continuous(0.6, 1, distribution='log-uniform') ,
                                   'num_leaves': Integer(10, 40),
                                   'learning_rate' : Continuous(0.01, 0.2, distribution='log-uniform'),
                                   'min_child_samples' : Integer(10, 40)
                                   }
                         
                         }
        elif self.mode == 'bayes':
            self.rf = {'model': ExtraTreesClassifier(n_jobs=self.n_jobs, n_estimators=100),
                       'param': {
                                 'max_depth': space.Integer(2, 10),
                                 'min_samples_split' : space.Integer(2, 20),
                                 'min_samples_leaf' : space.Integer(2,20),
                       }}
            
            

            self.lg = {'model': LogisticRegression(max_iter=1000, penalty='elasticnet', solver='saga'),
                       'param': {"C": space.Real(1e-4, 1e3, prior='log-uniform'),
                                 "l1_ratio": space.Real(0, 1)}}

            self.svc = {'model': SVC(probability=True),
                        'param': {'kernel': space.Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                                  'gamma': space.Real(1e-4, 1e4, prior='log-uniform'),
                                  'C': space.Real(1e-4, 1e4, prior='log-uniform'),
                                  'degree': space.Integer(1, 6)}
                       }

            self.xgb = {'model': XGBClassifier(n_jobs=self.n_jobs, eval_metric='logloss',
                                               use_label_encoder=False, n_estimators=100),
                        'param': {'learning_rate' : space.Real(0.01, 0.2, prior='log-uniform'),
                                  'max_depth': space.Integer(2, 10),
                                  'reg_alpha': space.Real(1e-3, 1, prior='log-uniform'),
                                  'reg_lambda': space.Real(1e-3, 1, prior='log-uniform'),
                                  'subsample' : space.Real(0.6, 1, prior='log-uniform'),
                                  'min_child_weight': space.Integer(1, 5),
                                  }
                        }
            
                                  

            self.lgbm = {'model': LGBMClassifier(n_jobs=self.n_jobs, n_estimators=100, objective='binary', max_depth=-1),
                         'param': {
                                   
                                   'subsample' : space.Real(0.6, 1, prior='log-uniform'),
                                   'learning_rate' : space.Real(0.01, 0.2, prior='log-uniform'),
                                   'num_leaves': space.Integer(10, 40),
                                    'min_child_samples' : space.Integer(10,40)
                                   }
                         }
                       
                        

           
        else:
            self.rf = {'model': ExtraTreesClassifier(n_jobs=self.n_jobs),
                       'param': {#'n_estimators': [100],
                           'n_estimators': range(10,30,2),
                                 #'max_depth': [10,20,30,40,50, None],
                           'max_depth': range(10,20,1),
                           #      'min_samples_split' : [2,5,7],
                           'min_samples_split' : range(2,5,1),
                                 #'min_samples_leaf' : [1,2,4]
                       }
                       }
            
            self.lg = {'model': LogisticRegression(),
                       'param': {"penalty": ['elasticnet'], "C": np.logspace(-4, 4, 20),
                                 "max_iter" : [1000], 'random_state': [1],
                                 "l1_ratio" : np.linspace(0,1, 10),
                                 'solver' : ['saga']} }
            #self.lg = {'model': LogisticRegression(max_iter=1000),
            #           'param': {"penalty": ["l2"], "C": np.logspace(-4, 4, 20)}}

            self.svc = {'model': SVC(),
                        'param': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                  'gamma': [100,10, 1, 0.1, 0.01, 0.001],
                                  'C': [0.01, 0.1, 1, 10, 100, 1000],
                                  'degree': [1, 2, 3, 4, 5, 6],
                                  'probability': [True]}
                        }

            self.xgb = {'model': XGBClassifier(n_jobs=self.n_jobs, eval_metric='logloss',
                                               use_label_encoder=False),
                        'param': {
                                  #'n_estimators': [100],
                                'n_estimators': np.arange(5,30,5),
                                  #'learning_rate' : [0.01, 0.1],
                                  #'max_depth': np.arange(2, 12, 2),
                            'max_depth': np.arange(2, 11, 1),
                                  'reg_alpha': [1e-2, 3.16e-2, 1e-1],
                                  'reg_lambda': [0.1, 0.316, 1],
                                  }
                        }

            self.lgbm = {'model': LGBMClassifier(n_jobs=self.n_jobs, objective='binary'),
                         'param': {#'n_estimators': [200],
                                 'n_estimators': np.arange(5,30,5),
                                   #'learning_rate' : [0.01, 0.1],
                                   #'num_leaves': np.arange(30, 50, 5),
                                   #'max_depth' : [-1],
                                   'max_depth' : np.arange(2,11,1),
                                   'num_leaves': np.arange(5, 25, 5),
                                   #'min_child_samples':[10,20,30,40],
                                 #  'reg_alpha': [1e-2, 3.16e-2, 1e-1],
                                 # 'reg_lambda': [0.1, 0.316, 1],
                                   }
                         }

        self.model_name = ['LGBM', 'XGB', 'RF', 'SVM', 'LG']
        self.models = [self.lgbm, self.xgb, self.rf, self.svc, self.lg]

    def _train(self, X_train, y_train):

        fitted_models = []
        scores = []
        st = time.time()
        for idx, model in enumerate(self.models):
            start_time = time.time()
            if self.mode == 'genetic':

                optim = GASearchCV(estimator=model['model'],
                                   param_grid=model['param'],
                                   n_jobs=self.n_jobs,
                                   verbose=False,
                                   scoring='accuracy',
                                   generations=40,
                                   cv=self.cv)
                callbacks = [TimerStopping(total_seconds=150), ConsecutiveStopping(generations=8, metric='fitness'),
                             ProgressBar()]
                optim.fit(X_train, y_train, callbacks=callbacks)

            elif self.mode == 'bayes':
                optim = BayesSearchCV(
                    model['model'],
                    model['param'],
                    n_iter=50,
                    scoring='accuracy',
                    n_jobs=self.n_jobs,
                    cv=self.cv
                )
                optim.fit(X_train, y_train)

            else:
                optim = GridSearchCV(estimator=model['model'],
                                     param_grid=model['param'],
                                     verbose=0,
                                     cv=self.cv,
                                     scoring='accuracy',
                                     n_jobs=self.n_jobs)
                optim.fit(X_train, y_train)

            
            estimator = model['model'].set_params(**optim.best_params_)
            joblib.dump(estimator, f'result/{self.now}/model/{self.pair}/{self.model_name[idx]}_{self.pair}.pkl')
            fitted_models.append(estimator)
            scores.append(optim.best_score_)
            print(self.model_name[idx] + f' 소요시간 {time.time() - start_time} cv정확도 {optim.best_score_}')
        print(f"**pair 개별 Estimator 최적화 소요시간 {round((time.time() - st) / 60, 2)}분")
        fitted_models = list(zip(self.model_name, fitted_models))
        with open(f'result/{self.now}/model/{self.pair}/estimators_scores_{self.pair}.pkl', 'wb') as f:
            pickle.dump(scores, f)
        return fitted_models, scores

    def _ensemble(self, X_train, y_train):
        fitted_models, scores = self._train(X_train, y_train)
        models_comb = list(combinations(fitted_models, 5)) + list(combinations(fitted_models, 4)) + list(combinations(
            fitted_models, 3)) + list(combinations(fitted_models, 2))
        models_scores = list(combinations(scores, 5)) + list(combinations(scores, 4)) + list(combinations(
            scores, 3)) + list(combinations(scores, 2))
        
        # voting_est = []
        # te_acc = 0
        # for idx, target_model in enumerate(fitted_models):
        #     voting_est.append(target_model)
        #     voting_clf = VotingClassifier(estimators=voting_est, voting='soft)
        #     voting_clf.fit(X_train, y_train)
        #     y_train_pred = voting_clf.predict(X_train)
        #     y_test_pred = voting_clf.predict(X_test)
        #     tr_acc = accuracy_score(y_train_pred, y_train)
        #     running_te_acc = accuracy_score(y_test_pred, y_test)
        #     if (running_te_acc > te_acc):
        #         result = voting_est
        #         te_acc = running_te_acc

        cv_acc = 0
        for idx, target_model in enumerate(models_comb):
            voting_clf = VotingClassifier(estimators=target_model, voting='soft', weights=models_scores[idx])
            cv_score = cross_val_score(voting_clf, X_train, y_train, scoring='accuracy', cv=self.cv)
            running_cv_score = np.mean(cv_score)
            
            result = target_model
            result_weights = models_scores[idx]
            if (running_cv_score > cv_acc) :
                cv_acc = running_cv_score
                result = target_model
                result_weights = models_scores[idx]
        
        voting_clf = VotingClassifier(estimators=result, weights=result_weights,
                                      voting='soft', n_jobs=self.n_jobs)
        voting_clf.fit(X_train, y_train)
        print(f"**pair Voting CV ACC {cv_acc}")
        joblib.dump(voting_clf, f'result/{self.now}/model/{self.pair}/model_before_fs_{self.pair}.pkl')
        
        
        return voting_clf

    def genetic_search_for_feature_selection(self, model, X_train, y_train):
        selector = GeneticSelectionCV(model,
                                  cv=self.cv,
                                  verbose=True,
                                  scoring="accuracy",
                                  n_population=50,
                                  max_features=len(X_train.columns) // 2,
                                  n_generations=50,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  n_gen_no_change=10,
                                  n_jobs=self.n_jobs,
                                  caching=True
                                  )
        selector = selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_]
        return selector.estimator_, selected_features
    
    def __call__(self, pairs, dataset):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.now= now
        os.makedirs(f'result/{self.now}/model', exist_ok=True)
        final_models = {}
        te_acc_ls = []
        start = time.time()
        for i in range(len(pairs)):
            
            self.pair = str(pairs[i])
            os.makedirs(f'result/{self.now}/model/{self.pair}', exist_ok=True)
            print(self.pair + ' 학습 시작')

            encoder = LabelEncoder()
            X_data, y_data = dataset[pairs[i]]
            if self.fs_mode != 'genetic':
                X_data = feature_select_by_vif(X_data, self.vif_thresholds)
                with open(f'result/{self.now}/model/{self.pair}/features_after_vif_{self.vif_thresholds}.pkl', 'wb') as f:
                    pickle.dump(X_data.columns.tolist(), f)

            X_train = X_data[:'2017']
            y_train = y_data[:'2017']
            y_train = encoder.fit_transform(y_train)

            X_test = X_data['2018':]
            y_test = y_data['2018':]
            y_test = encoder.transform(y_test)
            
            voting_clf = self._ensemble(X_train, y_train)
            
            if self.fs_mode == 'genetic':
                start_fs = time.time()
                voting_clf, selected_features = self.genetic_search_for_feature_selection(voting_clf, X_train, y_train)
                X_data = X_data[selected_features]
                print('*'+self.pair + f'피처 사이즈 {len(selected_features)} 변수탐색에 {round((time.time() - start_fs) / 60, 2)}분 소요')
                
            else:
                save_dir = (f'result/{self.now}/model/{self.pair}', self.pair)
                selected_features, features_mi, features_pi = post_hoc_feature_select(voting_clf, X_train, y_train, save_dir, self.mi, self.pi)
                with open(save_dir[0] + f'/features_from_MI_{self.pair}.pkl', 'wb') as f:
                    pickle.dump(features_mi, f)
                with open(save_dir[0] + f'/features_from_PI_{self.pair}.pkl', 'wb') as f:
                    pickle.dump(features_pi, f)
                with open(save_dir[0] + f'/features_final_{self.pair}.pkl', 'wb') as f:
                    pickle.dump(selected_features, f)
                X_data = X_data[selected_features]
                voting_clf.fit(X_data[:'2017'], y_train)
                
            
            y_test_pred = voting_clf.predict(X_data['2018':])
            te_acc = accuracy_score(y_test_pred, y_test)
            print(f'***테스트 정확도 {te_acc}')
            te_acc_ls.append((pairs[i], te_acc))
            y_test_pred = encoder.inverse_transform(y_test_pred)
            final_models[str(pairs[i])] = [voting_clf, y_test_pred, selected_features, encoder.classes_]
            joblib.dump(voting_clf, f'result/{self.now}/model/{self.pair}/model_{pairs[i]}.pkl')

        
        with open(f'result/{self.now}/model/final.pkl', 'wb') as f:
            pickle.dump(final_models, f)
           
        print(f"**전체학습 종료 : {round((time.time() - start) / 60, 2)}분 소요")
        print(f"테스트 정확도 : {te_acc_ls}")
        with open(f'result/{self.now}/model/test_accuracy.pkl', 'wb') as f:
            pickle.dump(te_acc_ls, f)
        return final_models, self.now
