# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# 1. 加载数据集
dataset = load_dataset("trl-lib/tldr", split="train")

# 2. 定义奖励函数（示例：距离 20 字符越近奖励越高）
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

# 3. 配置训练参数
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO",
    bf16=False,
    fp16=False,
)

# 4. 初始化并运行 Trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
