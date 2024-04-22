from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.special import softmax
from sklearn.ensemble import StackingClassifier
import numpy as np


# 여러 Excel 파일 로드 및 합치기
file_paths = glob.glob('./content/xlsx/Testfile*.xlsx') + glob.glob('./content/xlsx/generated_data_*.xlsx')
dfs = [pd.read_excel(fp) for fp in file_paths]  # 각 파일을 읽어서 데이터프레임 리스트 생성
df = pd.concat(dfs, ignore_index=True)  # 데이터프레임들을 하나로 합치기

df_sampled = df.sample(frac=0.1, random_state=42)

#print(df_sampled.columns)
#print(df_sampled.head())
#print(df_sampled.shape[0])

num_columns = df_sampled.shape[0]
df_sampled.reset_index(drop=True, inplace=True)

num_rows = 9

for i in range(1, num_rows + 1):
    col_name = f'VOLT {i}'
    diff_col_name = f'DIFF {col_name}'
    # 새로운 열을 미리 생성합니다. 초기값은 NaN 또는 0일 수 있습니다.
    df_sampled[diff_col_name] = 0.0
#    print(df_sampled[diff_col_name])
    
    for j in range(1, df_sampled.shape[0]):
        # .loc를 사용하여 값을 안전하게 할당
        df_sampled.loc[j, diff_col_name] = df_sampled.loc[j-1, col_name] - df_sampled.loc[j, col_name]

diff_volt_columns = [f'DIFF VOLT {i}' for i in range(1, num_rows + 1)]
df_sampled['Fault'] = df_sampled[diff_volt_columns].max(axis=1) >= 0.85


# 어떤 VOLT가 고장났는지 표시
def find_faulty_volts(row):
    return [i for i, val in enumerate(row[diff_volt_columns], 1) if val >= 0.85]

# 고장난 모선 검증
df_sampled['Faulty Volts'] = df_sampled.apply(find_faulty_volts, axis=1)

# volt1 - volt9 평균값 
df_sampled['VOLT_mean'] = df_sampled[[f'VOLT {i}' for i in range(1, num_rows + 1)]].mean(axis=1)

# volt1 - volt 9 표준편차
df_sampled['VOLT_std'] = df_sampled[[f'VOLT {i}' for i in range(1, num_rows + 1)]].std(axis=1)

# Softmax 함수
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Stability improvement by subtracting max from x
    return e_x / e_x.sum()

# SoftMax 열 추가 함수
#def add_softmax_columns(df, num_rows):
    # 각 VOLT 열에 대해 SoftMax 적용
#    for i in range(1, num_rows + 1):
#        volt_column = f'VOLT {i}'
#        softmax_volt_column = f'SoftMax VOLT {i}'
#        df[softmax_volt_column] = softmax(df[[volt_column]].values)  # 한 열에 대해서만 SoftMax 적용

    # 각 DIFF VOLT 열에 대해 SoftMax 적용
#    for i in range(1, num_rows + 1):
#        diff_volt_column = f'DIFF VOLT {i}'
#        softmax_diff_volt_column = f'SoftMax DIFF VOLT {i}'
#        df[softmax_diff_volt_column] = softmax(df[[diff_volt_column]].values)  # 한 열에 대해서만 SoftMax 적용

#    return df


# 새로운 열 추가
#df_sampled = add_softmax_columns(df_sampled, num_rows)

# Fault가 True인 경우만 필터링 및 추가 계산
faulty_rows = df_sampled[df_sampled['Fault'] == True]
faulty_rows['Failure_Probability'] = faulty_rows['VOLT_mean']**2 / faulty_rows['VOLT_std']

# 결과 출력, VOLT 1에서 VOLT 9, DIFF VOLT 1에서 DIFF VOLT 9
columns_to_show = ['Time(s)'] + [f'VOLT {i}' for i in range(1, num_rows + 1)] + [f'DIFF VOLT {i}' for i in range(1, num_rows + 1)] + ['Fault', 'Faulty Volts']
print(faulty_rows[columns_to_show])

columns_to_show2 = ['Time(s)'] + [f'DIFF VOLT {i}' for i in range(1, num_rows + 1)] + ['Fault', 'Faulty Volts']
print(faulty_rows[columns_to_show2])

print(faulty_rows[['Failure_Probability', 'VOLT_mean', 'VOLT_std']])

# 클래스 불균형 정도 확인
print(df_sampled['Fault'].value_counts())
print(df_sampled)




# 훈련 데이터와 테스트 데이터로 분할


X = df_sampled.drop(['Fault', 'Faulty Volts'], axis=1)  # 비특성 열 제거
y = df_sampled['Fault'].astype(int)  # Fault 열을 정수형으로 변환

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 첫 번째 열과 두 번째 열의 값 출력
print("First column values:\n", X.iloc[:, 0])
print("Second column values:\n", X.iloc[:, 1])

# 첫 번째 행과 두 번째 행의 값 출력
print("First row values:\n", X.iloc[0, :])
print("Second row values:\n", X.iloc[1, :])



# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE와 RandomUnderSampler 적용
# smote = SMOTE(random_state=42)
# rus = RandomUnderSampler(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
# X_train_res, y_train_res = rus.fit_resample(X_train_res, y_train_res)

# GBM°ú Random Forest ¸ðµ¨ ¼³Á¤ ¹× ·£´ý ¼­Ä¡
gbm = GradientBoostingClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42, probability=True)  # SVM 모델에 probability=True 설정


gbm_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [1,2,3],
    'max_features': ['sqrt', 'log2', None]  # 정규화 매개변수 'None' 추가
}


rf_param_dist = {
    'n_estimators': [300, 500, 700, 1000],
    'max_depth': [1, 2, 3, 4, 5, 6, 7],
    'max_features': ['sqrt', 'log2', None]
}


svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']  # 'auto' 옵션 추가
}

gbm_search = GridSearchCV(gbm, gbm_param_grid, cv=10, scoring='f1', n_jobs=-1)
#gbm_random_search = RandomizedSearchCV(gbm, gbm_param_dist, cv=5, scoring='f1', n_iter=20, random_state=42)
rf_random_search = RandomizedSearchCV(rf, rf_param_dist, cv=5, scoring='f1', n_iter=20, random_state=42)
#svm_random_search = RandomizedSearchCV(svm, svm_param_dist, cv=5, scoring='f1', n_iter=20, random_state=42)
svm_search = GridSearchCV(svm, svm_param_grid, cv=10, scoring='f1', n_jobs=-1)

#gbm_random_search.fit(X_train_res, y_train_res)
#gbm_search.fit(X_train_res, y_train_res)
gbm_search.fit(X_train_scaled, y_train)
#rf_random_search.fit(X_train_res, y_train_res)
rf_random_search.fit(X_train_scaled, y_train)
#svm_random_search.fit(X_train_res, y_train_res)
#svm_search.fit(X_train_res, y_train_res)
svm_search.fit(X_train_scaled, y_train)


# 모델 평가
# gbm_y_pred = gbm_random_search.best_estimator_.predict(X_test_scaled)
gbm_y_pred = gbm_search.best_estimator_.predict(X_test_scaled)
rf_y_pred = rf_random_search.best_estimator_.predict(X_test_scaled)
# svm_y_pred = svm_random_search.best_estimator_.predict(X_test_scaled)
svm_y_pred = svm_search.best_estimator_.predict(X_test_scaled)

gbm_y_pred_proba = gbm_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]  # Get the probability of the positive class
rf_y_pred_proba = rf_random_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]
svm_y_pred_proba = svm_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]

# Applying softmax to the output probabilities of each model
gbm_y_pred_proba_softmax = softmax([gbm_y_pred_proba])
rf_y_pred_proba_softmax = softmax([rf_y_pred_proba])
svm_y_pred_proba_softmax = softmax([svm_y_pred_proba])


# 예측 결과에 Time(s) 열 추가
gbm_y_pred_with_time = pd.DataFrame({'Time(s)': X_test['Time(s)'], 'Fault': gbm_y_pred})
rf_y_pred_with_time = pd.DataFrame({'Time(s)': X_test['Time(s)'], 'Fault': rf_y_pred})
svm_y_pred_with_time = pd.DataFrame({'Time(s)': X_test['Time(s)'], 'Fault': svm_y_pred})

time_bins = pd.qcut(gbm_y_pred_with_time['Time(s)'], q=10, duplicates='drop')
time_bins = pd.qcut(rf_y_pred_with_time['Time(s)'], q=10, duplicates='drop')
time_bins = pd.qcut(svm_y_pred_with_time['Time(s)'], q=10, duplicates='drop')



print("GBM Metrics:")
print("Precision:", precision_score(y_test, gbm_y_pred))
print("Recall:", recall_score(y_test, gbm_y_pred))
print("F1-score:", f1_score(y_test, gbm_y_pred))
print("ROC-AUC:", roc_auc_score(y_test, gbm_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]))
print("Accuracy:", accuracy_score(y_test, gbm_y_pred))
# print("GBM Probabilities after Softmax:", gbm_y_pred_proba_softmax)
print("GBM Probability of Predicting 1 after Softmax:", gbm_y_pred_proba_softmax[0][1])
# print("GBM Fault Probability by Time Bin:",gbm_y_pred_with_time.groupby(time_bins)['Fault'].mean())


print("\nRandom Forest Metrics:")
print("Precision:", precision_score(y_test, rf_y_pred))
print("Recall:", recall_score(y_test, rf_y_pred))
print("F1-score:", f1_score(y_test, rf_y_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_random_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]))
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
# print("RF Probabilities after Softmax:", rf_y_pred_proba_softmax)
print("RF Probability of Predicting 1 after Softmax:", rf_y_pred_proba_softmax[0][1])
# print("\nRandom Forest Fault Probability by Time Bin:", rf_y_pred_with_time.groupby(time_bins)['Fault'].mean())

print("\nSVM Metrics:")
print("Precision:", precision_score(y_test, svm_y_pred))
print("Recall:", recall_score(y_test, svm_y_pred))
print("F1-score:", f1_score(y_test, svm_y_pred))
print("ROC-AUC:", roc_auc_score(y_test, svm_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]))
print("Accuracy:", accuracy_score(y_test, svm_y_pred))
# print("SVM Probabilities after Softmax:", svm_y_pred_proba_softmax)
print("SVM Probability of Predicting 1 after Softmax:", svm_y_pred_proba_softmax[0][1])
# print("\nSVM Fault Probability by Time Bin:", svm_y_pred_with_time.groupby(time_bins)['Fault'].mean())


estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# 스태킹 모델 정의, 메타 모델로 SVC 사용
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=SVC(probability=True, random_state=42)
)

# 스태킹 모델 훈련
#stack_model.fit(X_train_res, y_train_res)
stack_model.fit(X_train_scaled, y_train)

# 스태킹 모델을 사용한 예측
y_pred = stack_model.predict(X_test_scaled)
y_pred_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

# 예측 결과 평가
print("\nStacking Metrics:")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("Accuracy:", accuracy_score(y_test, y_pred))