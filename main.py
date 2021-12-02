import sys
import warnings
import os
import random
import numpy as np
from sklearn.exceptions import ConvergenceWarning
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

from utils.config import global_setting
from utils.dataload import load_dataset
from utils.magi_module import get_report
from utils.models import Ensemble
from utils.result import get_score_table
from utils.utils import stationary_transform



def main(start_seed=0, end_seed=10, mode='grid', fs_mode='stat', vif=3000, mi=True, pi=True):
    # 데이터 로드
    total_dataset = load_dataset()
    pairs = list(total_dataset.keys())
    #total_dataset = stationary_transform(total_dataset, pairs)

    for seed in range(start_seed, end_seed):
        print("Seed :", seed)
        
        # 전역 환경설정
        global_setting()
        random.seed(seed)
        np.random.seed(seed)

        # 모델학습
        #try:
        model = Ensemble(mode=mode, fs_mode=fs_mode, vif=vif, mi=mi, pi=pi)
        final_models, now = model(pairs, total_dataset)
        #except:
         #   continue

        # 결과 저장
        result = get_score_table(total_dataset, final_models, now)
        submission_score = result.set_index(['tdate', 'code'])['score'].unstack()
        summary, ress = get_report(submission_score)        
        ress.plot(figsize=(10, 5))
        ress.to_csv(f'result/{now}/score/ress.csv')
        summary.to_csv(f'result/{now}/score/summary.csv')

if __name__ == '__main__':
    main(start_seed, end_seed, mode='grid', fs_mode='stats', vif=3000, mi=True, pi=False)