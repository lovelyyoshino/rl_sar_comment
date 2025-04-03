# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pandas as pd

BASE_PATH = os.path.join(os.path.dirname(__file__), "../")

class Config:
    def __init__(self):
        # 学习率
        self.lr = 8e-4
        # 数值稳定性常数
        self.eps = 1e-8
        # 权重衰减
        self.weight_decay = 0.0
        # 训练的轮数
        self.epochs = 200
        # 每个批次的大小
        self.batch_size = 128
        # 设备类型（CPU或GPU）
        self.device = "cuda:0"
        # 输入维度
        self.in_dim = 6
        # 隐藏层单元数
        self.units = 32
        # 隐藏层数量
        self.layers = 2
        # 输出维度
        self.out_dim = 1
        # 激活函数类型
        self.act = "softsign"
        # 时间步长
        self.dt = 0.02

class ActuatorDataset(Dataset):
    def __init__(self, data):
        # 初始化数据集
        self.data = data

    def __len__(self):
        # 返回数据集中关节状态的数量
        return len(self.data["joint_states"])

    def __getitem__(self, idx):
        # 根据索引返回数据字典中的一项
        return {k: v[idx] for k, v in self.data.items()}

class Act(nn.Module):
    def __init__(self, act, slope=0.05):
        super(Act, self).__init__()
        # 激活函数类型
        self.act = act
        # 斜率，用于某些激活函数
        self.slope = slope
        # 计算softplus的偏移量
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, input):
        # 根据指定的激活函数类型进行前向传播
        if self.act == "relu":
            return F.relu(input)
        elif self.act == "leaky_relu":
            return F.leaky_relu(input)
        elif self.act == "sp":
            return F.softplus(input, beta=1.0)
        elif self.act == "leaky_sp":
            return F.softplus(input, beta=1.0) - self.slope * F.relu(-input)
        elif self.act == "elu":
            return F.elu(input, alpha=1.0)
        elif self.act == "leaky_elu":
            return F.elu(input, alpha=1.0) - self.slope * F.relu(-input)
        elif self.act == "ssp":
            return F.softplus(input, beta=1.0) - self.shift
        elif self.act == "leaky_ssp":
            return (
                F.softplus(input, beta=1.0) - self.slope * F.relu(-input) - self.shift
            )
        elif self.act == "tanh":
            return torch.tanh(input)
        elif self.act == "leaky_tanh":
            return torch.tanh(input) + self.slope * input
        elif self.act == "swish":
            return torch.sigmoid(input) * input
        elif self.act == "softsign":
            return F.softsign(input)
        else:
            raise RuntimeError(f"Undefined activation called {self.act}")

def build_mlp(config):
    # 构建多层感知机（MLP）模型，这一行代表了线性层，然后激活函数层
    mods = [nn.Linear(config.in_dim, config.units), Act(config.act)]
    for i in range(config.layers - 1):
        mods += [nn.Linear(config.units, config.units), Act(config.act)]
    mods += [nn.Linear(config.units, config.out_dim)]
    return nn.Sequential(*mods)

def load_data(data_path):
    # 从CSV文件加载数据
    data = pd.read_csv(data_path)
    if len(data) < 1:
        return None, 0

    # 统计电机数量，统计tau_est_的列数
    num_motors = sum(1 for col in data.columns if col.startswith("tau_est_"))
    columns = ["tau_est_", "tau_cal_", "joint_pos_", "joint_pos_target_", "joint_vel_"]

    # 初始化数据字典
    data_dict = {col: [] for col in columns}
    for col in columns:
        for i in range(num_motors):
            data_dict[col].append(data[f"{col}{i}"].values)

    # 转换为numpy数组并转置
    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key]).T

    return data_dict, num_motors

def process_data(data_dict, num_motors, step):
    # 处理数据以生成输入和输出，这里需要拿到joint_position_errors、joint_velocities、tau_ests
    joint_position_errors = data_dict["joint_pos_target_"] - data_dict["joint_pos_"]
    joint_velocities = data_dict["joint_vel_"]
    tau_ests = data_dict["tau_est_"]

    # 将数据转换为张量
    joint_position_errors = torch.tensor(joint_position_errors, dtype=torch.float)
    joint_velocities = torch.tensor(joint_velocities, dtype=torch.float)
    tau_ests = torch.tensor(tau_ests, dtype=torch.float)

    xs, ys = [], []
    for i in range(num_motors):#我们写入的时候是12列
        # 为每个电机构建输入特征，这里取传入的数据是12列，所以i取0-11
        # 然后前面的step是2，这代表位置误差的当前、前一和前两步（3个位置误差）
        xs_joint = [
            joint_position_errors [step:    , i:i+1],
            joint_position_errors [step-1:-1, i:i+1],
            joint_position_errors [step-2:-2, i:i+1],
            joint_velocities      [step:    , i:i+1],
            joint_velocities      [step-1:-1, i:i+1],
            joint_velocities      [step-2:-2, i:i+1],
        ]#这里6维向量 (3个位置误差和3个速度值)，对应的其实就是真实的运动电机位置，以及目标位置，以及目标速度
        tau_ests_joint = tau_ests[step:    , i:i+1]# 输出：1维扭矩值

        # 合并特征并添加到列表
        xs_joint = torch.cat(xs_joint, dim=1)
        xs.append(xs_joint)
        ys.append(tau_ests_joint)

    # 合并所有电机的输入和输出
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    return xs, ys

def train_actuator_network(xs, ys, actuator_network_path, config):
    # 训练执行器网络
    num_data = xs.shape[0]
    num_train = num_data // 5 * 4
    num_test = num_data - num_train

    # 创建数据集并分割为训练集和验证集
    dataset = ActuatorDataset({"joint_states": xs, "tau_ests": ys})#这里需要传入一个字典，字典的键是joint_states，值是xs，tau_ests是ys
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_test])#列表中是训练集和验证集的比例
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)

    # 构建模型
    model = build_mlp(config)

    # 定义优化器
    opt = Adam(model.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)

    # 将模型移动到指定设备
    model = model.to(config.device)
    for epoch in range(config.epochs):
        epoch_loss = 0
        ct = 0
        for batch in train_loader:
            data = batch["joint_states"].to(config.device)
            y_pred = model(data)

            opt.zero_grad()

            y_label = batch["tau_ests"].to(config.device)

            # 计算损失
            tau_est_loss = ((y_pred - y_label) ** 2).mean()
            loss = tau_est_loss

            loss.backward()
            opt.step()
            epoch_loss += loss.detach().cpu().numpy()
            ct += 1
        epoch_loss /= ct

        test_loss = 0
        mae = 0
        ct = 0
        if epoch % 1 == 0:
            with torch.no_grad():
                for batch in test_loader:
                    data = batch["joint_states"].to(config.device)
                    y_pred = model(data)

                    y_label = batch["tau_ests"].to(config.device)

                    # 计算测试损失
                    tau_est_loss = ((y_pred - y_label) ** 2).mean()
                    loss = tau_est_loss
                    test_mae = (y_pred - y_label).abs().mean()

                    test_loss += loss
                    mae += test_mae
                    ct += 1
                test_loss /= ct
                mae /= ct

            print(f"epoch: {epoch} | loss: {epoch_loss:.4f} | test loss: {test_loss:.4f} | mae: {mae:.4f}")

        # 导出模型为TorchScript
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(actuator_network_path)  # Save
    return model

def train_actuator_network_and_plot_predictions(data_path, actuator_network_path, load_pretrained_model=False, config=None):
    # 训练执行器网络并绘制预测结果
    print(f"Load data: {data_path}")

    data_dict, num_motors = load_data(data_path)
    if data_dict is None:
        print(f"Failed to load data from {data_path}")
        return
    step = 2
    xs, ys = process_data(data_dict, num_motors, step)

    if load_pretrained_model:
        print("Evaluating the existing actuator network...")
        model = torch.jit.load(actuator_network_path).to("cpu")
        print(f"Use trained actuator network model: {actuator_network_path}")
    else:
        print("Training a new actuator network...")
        model = train_actuator_network(xs, ys, actuator_network_path, config).to("cpu")
        print(f"Saving actuator network model to: {actuator_network_path}")

    # 进行预测，这里使用num_motors代表关节数量，xs代表输入，model代表模型，detach代表不计算梯度，reshape代表重塑形状，T代表转置
    tau_preds = model(xs).detach().reshape(num_motors, -1).T

    plot_length = 1000

    # 生成时间步长
    timesteps = np.array(range(len(data_dict["tau_est_"]))) * config.dt
    timesteps = timesteps[step:step + len(tau_preds)]

    tau_cals = data_dict["tau_cal_"][step:step + len(tau_preds)]
    tau_ests = data_dict["tau_est_"][step:step + len(tau_preds)]
    tau_preds = tau_preds[:plot_length]

    # 绘制结果
    fig, axs = plt.subplots(6, 2, figsize=(14, 6))
    axs = np.array(axs).flatten()
    for i in range(num_motors):#将真实和预测的扭矩测算出来，看看对比效果
        axs[i].plot(timesteps[:plot_length], tau_cals[:plot_length, i], label="Calculated torque")
        axs[i].plot(timesteps[:plot_length], tau_ests[:plot_length, i], label="Real torque")
        axs[i].plot(timesteps[:plot_length], tau_preds[:plot_length, i], label="Predicted torque", linestyle="--")
    fig.legend(["Calculated torque", "Real torque", "Predicted torque"], loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()

def main():
    # 主函数，解析命令行参数并调用训练或评估函数
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "play"], help="Choose whether to train or evaluate the actuator network")
    parser.add_argument("--data", type=str, required=True, help="Path of data files")
    parser.add_argument("--output", type=str, required=True, help="Path to save or load the actuator network model")

    args = parser.parse_args()

    data_path = os.path.join(BASE_PATH, "models", args.data)
    output_path = os.path.join(BASE_PATH, "models", args.output)

    config = Config()

    if args.mode == "train":
        load_pretrained_model = False
    elif args.mode == "play":
        load_pretrained_model = True

    # 调用训练和绘图函数
    train_actuator_network_and_plot_predictions(
        data_path=data_path,
        actuator_network_path=output_path,
        load_pretrained_model=load_pretrained_model,
        config=config,
    )

if __name__ == "__main__":
    main()
