import joblib
import optuna
from optuna.storages import RDBStorage
'''将本地study文件加载到新的study中实现恢复 '''
# 加载要恢复的study的 .pkl 文件
study = joblib.load("./runs/optuna_study_teacher.pkl")

# 检查是否为 Study 对象
if isinstance(study, optuna.study.Study):
    trials = study.trials  # 获取所有 trial 数据
    print(f"Loaded {len(trials)} trials from .pkl file.")
else:
    print("The .pkl file does not contain a valid Study object.")
    exit()

# 要加载到的数据库路径
db_path = "sqlite:///./runs/optuna_study.db"
storage = RDBStorage(db_path)

# 创建新的 study
new_study = optuna.create_study(
    study_name="teacher_model_optimization_restored",
    storage=storage,
    direction="minimize",  # 之前是单一目标（MAE）
    load_if_exists=True
)

# 将 trial 数据添加到新 study 中
for trial in trials:
    new_study.add_trial(trial)

print(f"Added {len(trials)} trials to new study.")

# 打印新 study 的最佳 trial
best_trial = new_study.best_trial
print(f"Best trial number: {best_trial.number}")
print(f"Best value (MAE): {best_trial.value}")
print(f"Best params: {best_trial.params}")