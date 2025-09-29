import joblib
import optuna
from optuna.storages import RDBStorage
import argparse
import json
'''
将保存在本地 .pkl 文件中的 Optuna study 恢复并加载到 SQLite 数据库中。
这对于从中断的、未连接数据库的运行中恢复，或迁移历史运行记录非常有用。
'''

if __name__ == "__main__":
    # --- 新增: 使用 argparse 使脚本更灵活 ---
    parser = argparse.ArgumentParser(description="Restore Optuna study from a .pkl file to a SQLite database.")
    parser.add_argument("--pkl_file", '-f', type=str, required=True, help="Path to the .pkl file to restore.")
    parser.add_argument("--db_path", '-d', type=str, default="sqlite:///./runs/optuna_study.db", help="Path to the SQLite database file.")
    parser.add_argument("--study_name", '-n', type=str, required=True, help="The name for the new study in the database.")
    args = parser.parse_args()

    # 加载要恢复的 .pkl 文件
    try:
        study_to_restore = joblib.load(args.pkl_file)
        # 检查是否为 Study 对象
        if isinstance(study_to_restore, optuna.study.Study):
            trials = study_to_restore.trials
            print(f"Loaded {len(trials)} trials from '{args.pkl_file}'.")
        else:
            # 有时 .pkl 文件可能只保存了 best_trials 列表
            print(f"Warning: The .pkl file does not contain a full Study object. Trying to load a list of trials.")
            trials = study_to_restore
            if not isinstance(trials, list) or not all(isinstance(t, optuna.trial.FrozenTrial) for t in trials):
                 raise TypeError("The .pkl file does not contain a valid Study object or a list of FrozenTrial objects.")
            print(f"Loaded {len(trials)} trials from the list in '{args.pkl_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{args.pkl_file}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the .pkl file: {e}")
        exit()

    # --- 修改: 配置多目标优化的 directions ---
    # 这个列表必须与您在训练脚本中使用的 directions 完全一致
    study_directions = ["maximize", "minimize", "minimize", "minimize"]

    # 连接到数据库并创建新的 study
    storage = RDBStorage(args.db_path)
    new_study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=study_directions,  # 使用多目标 directions
        load_if_exists=True # 如果同名 study 已存在，则加载它
    )
    print(f"Successfully created or loaded study '{args.study_name}' in '{args.db_path}'.")

    # 将从 .pkl 文件中加载的 trial 数据逐个添加到新的 study 中
    added_trials_count = 0
    for trial in trials:
        try:
            new_study.add_trial(trial)
            added_trials_count += 1
        except Exception as e:
            # 如果试验已存在或有其他问题，则打印警告
            print(f"Warning: Could not add trial number {trial.number}. Reason: {e}")

    print(f"Successfully added {added_trials_count} new trials to the study '{args.study_name}'.")

    # 打印新 study 的 Pareto 前沿结果
    print("\n" + "="*50)
    print(f"           Best Trials (Pareto Front) for '{args.study_name}'")
    print("="*50)
    if new_study.best_trials:
        for trial in new_study.best_trials:
            print(f"Trial Number: {trial.number}")
            print(f"  - Values (MAIS Acc, HIC MAE, Dmax MAE, Nij MAE): {trial.values}")
            print(f"  - Params: {json.dumps(trial.params, indent=4)}")
            print("-" * 20)
    else:
        print("No best trials found yet.")