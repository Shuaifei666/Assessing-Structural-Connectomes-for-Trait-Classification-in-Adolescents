import pandas as pd

path_to_dataset1 = '/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/label/xtrm_label/fitbit_label.csv'
path_to_dataset2 = '/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/Fitbit_data/all_vectorized_data_fitbit.csv'
df1 = pd.read_csv(path_to_dataset1)
df2 = pd.read_csv(path_to_dataset2)

merged_df = pd.merge(df1, df2, on='ID', how='inner')

merged_df.to_csv('/nas/longleaf/home/yifzhang/zhengwu/565proj/psc_connect/final_data/merged_dataset_fitbit.csv', index=False)

