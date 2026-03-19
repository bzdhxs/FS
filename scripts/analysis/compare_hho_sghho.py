"""HHO vs SGHHO 对比脚本

自动运行 HHO 和 SGHHO 算法，分别配合 PLSR、SVM、RF 三种模型，
并生成对比结果的 Markdown 表格。

使用方法:
    python script/compare_hho_sghho.py

输出:
    - log/comparison_results.md: 对比结果表格
    - log/comparison.log: 详细运行日志
    - log/YYYYMMDD_HHMMSS_<ALGO>_<MODEL>/: 各实验的详细日志
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 实验配置
ALGORITHMS = ["HHO", "SGHHO"]
MODELS = ["PLS", "SVM", "RF"]
CONFIG_FILES = {
    "HHO": "config_hho.yaml",
    "SGHHO": "config_sghho_standard.yaml"
}


def setup_comparison_logger() -> logging.Logger:
    """设置对比脚本的日志记录器"""
    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "comparison.log"
    
    logger = logging.getLogger("comparison")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def run_experiment(algo: str, model: str, logger: logging.Logger) -> Tuple[bool, str]:
    """运行单个实验
    
    Args:
        algo: 算法名称 (HHO/SGHHO)
        model: 模型名称 (PLS/SVM/RF)
        logger: 日志记录器
    
    Returns:
        (成功标志, 日志目录路径)
    """
    config_file = CONFIG_FILES[algo]
    
    logger.info(f"{'='*60}")
    logger.info(f"开始实验: {algo} + {model}")
    logger.info(f"配置文件: {config_file}")
    logger.info(f"{'='*60}")
    
    # 修改配置文件中的模型名称
    config_path = PROJECT_ROOT / config_file
    
    try:
        # 读取配置
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 修改模型
        config['model_name'] = model
        
        # 临时保存配置
        temp_config = PROJECT_ROOT / f"temp_config_{algo}_{model}.yaml"
        with open(temp_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        # 运行实验
        cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), "--config", str(temp_config)]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # 删除临时配置
        temp_config.unlink()
        
        if result.returncode == 0:
            logger.info(f"✓ 实验成功: {algo} + {model}")
            
            # 查找最新的日志目录
            log_dir = PROJECT_ROOT / "log"
            pattern = f"*_{algo}_{model}"
            matching_dirs = sorted(log_dir.glob(pattern))
            
            if matching_dirs:
                latest_dir = matching_dirs[-1]
                logger.info(f"日志目录: {latest_dir.name}")
                return True, str(latest_dir)
            else:
                logger.warning(f"未找到日志目录: {pattern}")
                return False, ""
        else:
            logger.error(f"✗ 实验失败: {algo} + {model}")
            logger.error(f"错误信息: {result.stderr}")
            return False, ""
            
    except Exception as e:
        logger.error(f"✗ 实验异常: {algo} + {model}")
        logger.error(f"异常信息: {str(e)}")
        return False, ""


def extract_results(log_dir: str, logger: logging.Logger) -> Dict:
    """从日志目录提取结果
    
    Args:
        log_dir: 日志目录路径
        logger: 日志记录器
    
    Returns:
        结果字典
    """
    summary_file = Path(log_dir) / "summary.json"
    
    if not summary_file.exists():
        logger.warning(f"未找到 summary.json: {log_dir}")
        return None
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            'n_features': data['feature_selection']['n_selected'],
            'train_r2': data['performance']['train']['R2'],
            'train_rmse': data['performance']['train']['RMSE'],
            'train_mae': data['performance']['train'].get('MAE', 'N/A'),
            'test_r2': data['performance']['test']['R2'],
            'test_rmse': data['performance']['test']['RMSE'],
            'test_mae': data['performance']['test'].get('MAE', 'N/A'),
            'selected_indices': data['feature_selection'].get('selected_indices', [])
        }
    except Exception as e:
        logger.error(f"提取结果失败: {log_dir}")
        logger.error(f"错误: {str(e)}")
        return None


def generate_markdown_table(results: Dict[Tuple[str, str], Dict], logger: logging.Logger) -> str:
    """生成 Markdown 对比表格
    
    Args:
        results: 实验结果字典 {(algo, model): result_dict}
        logger: 日志记录器
    
    Returns:
        Markdown 表格字符串
    """
    lines = []
    lines.append("# HHO vs SGHHO 对比结果\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("")
    
    # 表头
    lines.append("| 特征筛选算法 | 模型 | 选择特征数 | 训练集R² | 训练集RMSE | 训练集MAE | 测试集R² | 测试集RMSE | 测试集MAE |")
    lines.append("|------------|------|-----------|---------|-----------|----------|---------|-----------|----------|")
    
    # 数据行
    for algo in ALGORITHMS:
        for model in MODELS:
            key = (algo, model)
            if key in results and results[key]:
                r = results[key]
                
                # 格式化 MAE（可能是 'N/A'）
                train_mae_str = f"{r['train_mae']:.4f}" if isinstance(r['train_mae'], (int, float)) else r['train_mae']
                test_mae_str = f"{r['test_mae']:.4f}" if isinstance(r['test_mae'], (int, float)) else r['test_mae']
                
                line = (
                    f"| {algo:10s} | {model:4s} | {r['n_features']:9d} | "
                    f"{r['train_r2']:7.4f} | {r['train_rmse']:9.4f} | {train_mae_str:8s} | "
                    f"{r['test_r2']:7.4f} | {r['test_rmse']:9.4f} | {test_mae_str:8s} |"
                )
                lines.append(line)
            else:
                lines.append(f"| {algo:10s} | {model:4s} | 失败      | -       | -         | -        | -       | -         | -        |")
    
    lines.append("")
    lines.append("## 详细信息\n")
    
    # 添加选择的特征索引
    for algo in ALGORITHMS:
        lines.append(f"### {algo} 选择的特征\n")
        for model in MODELS:
            key = (algo, model)
            if key in results and results[key]:
                r = results[key]
                indices = r.get('selected_indices', [])
                if indices:
                    lines.append(f"**{model}**: {len(indices)} 个特征")
                    lines.append(f"```")
                    lines.append(f"{indices}")
                    lines.append(f"```")
                else:
                    lines.append(f"**{model}**: 无特征索引信息")
            else:
                lines.append(f"**{model}**: 实验失败")
            lines.append("")
    
    return "\n".join(lines)


def main():
    """主函数"""
    logger = setup_comparison_logger()
    
    logger.info("="*60)
    logger.info("HHO vs SGHHO 对比实验开始")
    logger.info("="*60)
    logger.info(f"算法: {ALGORITHMS}")
    logger.info(f"模型: {MODELS}")
    logger.info(f"总实验数: {len(ALGORITHMS) * len(MODELS)}")
    logger.info("")
    
    # 存储所有实验结果
    all_results = {}
    
    # 运行所有实验
    for algo in ALGORITHMS:
        for model in MODELS:
            success, log_dir = run_experiment(algo, model, logger)
            
            if success and log_dir:
                result = extract_results(log_dir, logger)
                all_results[(algo, model)] = result
            else:
                all_results[(algo, model)] = None
            
            logger.info("")
    
    # 生成对比表格
    logger.info("="*60)
    logger.info("生成对比结果表格")
    logger.info("="*60)
    
    markdown_content = generate_markdown_table(all_results, logger)
    
    # 保存到文件
    output_file = PROJECT_ROOT / "log" / "comparison_results.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    logger.info(f"对比结果已保存: {output_file}")
    logger.info("")
    
    # 统计成功/失败
    total = len(ALGORITHMS) * len(MODELS)
    success_count = sum(1 for v in all_results.values() if v is not None)
    
    logger.info("="*60)
    logger.info("实验完成")
    logger.info("="*60)
    logger.info(f"成功: {success_count}/{total}")
    logger.info(f"失败: {total - success_count}/{total}")
    logger.info(f"结果文件: {output_file}")
    logger.info(f"日志文件: {PROJECT_ROOT / 'log' / 'comparison.log'}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
