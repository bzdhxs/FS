# utils/data_processor.py

import os
import pandas as pd
import logging

from utils.data_split import regression_stratified_split


class DataProcessor:
    """
    数据处理器：负责数据的读取、清洗和分层抽样。
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("DataProcessor")

    def load_and_preprocess(self,
                            original_data_path,
                            target_col,
                            output_dir,
                            test_size=0.3,
                            random_state=42):
        """
        执行完整流程：读取 -> ID检查 -> 分层抽样 -> 保存

        Args:
            original_data_path (str): 原始 CSV 路径
            target_col (str): 目标变量列名 (如 'TS')
            output_dir (str): 输出目录
            test_size (float): 测试集比例

        Returns:
            tuple: (train_file_path, test_file_path)
        """
        # 1. 定义输出路径 - 保存到 data 子目录
        data_dir = os.path.join(output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        train_file = os.path.join(data_dir, 'train.csv')
        test_file = os.path.join(data_dir, 'test.csv')

        # 2. 读取数据
        if not os.path.exists(original_data_path):
            self.logger.error(f"❌ 找不到原始数据: {original_data_path}")
            raise FileNotFoundError(f"{original_data_path} not found.")

        self.logger.info(f"   [Processor] 读取原始数据: {os.path.basename(original_data_path)}")
        df_full = pd.read_csv(original_data_path)

        # 3. 确保 Sample_ID 存在 (防止后续合并出错)
        if 'Sample_ID' not in df_full.columns:
            df_full = pd.concat(
                [pd.Series(df_full.index, name='Sample_ID'), df_full], axis=1
            )

        # 4. 分离特征与标签
        X = df_full.drop(columns=[target_col])
        y = df_full[target_col]

        # 5. 执行分层抽样 (Stratified Split)
        self.logger.info(f"   [Processor] 执行分层抽样 (Test Size: {test_size})...")
        X_train, X_test, y_train, y_test = regression_stratified_split(
            X, y, test_size=test_size, n_bins=5, random_state=random_state
        )

        # 6. 合并数据并保存
        df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)

        self.logger.info(f"   ✅ 数据预处理完成. Train: {len(df_train)}, Test: {len(df_test)}")

        return train_file, test_file
