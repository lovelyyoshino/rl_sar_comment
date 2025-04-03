/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

 #include "rl_real_a1.hpp"

 // #define PLOT
 // #define CSV_LOGGER
 
 // RL_Real类的构造函数，初始化机器人控制和参数
 RL_Real::RL_Real() : unitree_safe(UNITREE_LEGGED_SDK::LeggedType::A1), unitree_udp(UNITREE_LEGGED_SDK::LOWLEVEL)
 {
     // 从YAML文件读取参数
     this->robot_name = "a1"; // 机器人名称
     this->config_name = "legged_gym"; // 配置名称
     std::string robot_path = this->robot_name + "/" + this->config_name; // 机器人路径
     this->ReadYaml(robot_path); // 读取YAML配置
 
     // 修改观察参数中的角速度名称
     for (std::string &observation : this->params.observations)
     {
         // 在Unitree A1中，角速度的坐标系是机体坐标系
         if (observation == "ang_vel")
         {
             observation = "ang_vel_body"; // 修改为机体坐标系下的角速度
         }
     }
 
     // 初始化机器人
     this->unitree_udp.InitCmdData(this->unitree_low_command); // 初始化命令数据，将传入的command数据初始化赋值为0
 
     // 初始化强化学习
     torch::autograd::GradMode::set_enabled(false); // 禁用梯度计算
     torch::set_num_threads(4); // 设置线程数
     if (!this->params.observations_history.empty())
     {
         this->history_obs_buf = ObservationBuffer(1, this->params.num_observations, this->params.observations_history.size()); // 初始化观察历史缓冲区，设置观察关节和观察步长
     }
     this->InitObservations(); // 初始化观察
     this->InitOutputs(); // 初始化输出
     this->InitControl(); // 初始化控制
     running_state = STATE_WAITING; // 设置初始运行状态为等待
 
     // 加载模型
     std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + robot_path + "/" + this->params.model_name; // 模型路径
     this->model = torch::jit::load(model_path); // 加载TorchScript模型
 
     // 启动循环
     this->loop_udpSend = std::make_shared<LoopFunc>("loop_udpSend", 0.002, std::bind(&RL_Real::UDPSend, this), 3); // UDP发送循环
     this->loop_udpRecv = std::make_shared<LoopFunc>("loop_udpRecv", 0.002, std::bind(&RL_Real::UDPRecv, this), 3); // UDP接收循环
     this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this)); // 键盘输入循环
     this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this)); // 机器人控制循环
     this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this)); // 强化学习循环
     this->loop_udpSend->start(); // 启动UDP发送循环
     this->loop_udpRecv->start(); // 启动UDP接收循环
     this->loop_keyboard->start(); // 启动键盘输入循环
     this->loop_control->start(); // 启动机器人控制循环
     this->loop_rl->start(); // 启动强化学习循环
 
 #ifdef PLOT
     // 初始化绘图相关数据
     this->plot_t = std::vector<int>(this->plot_size, 0);
     this->plot_real_joint_pos.resize(this->params.num_of_dofs);
     this->plot_target_joint_pos.resize(this->params.num_of_dofs);
     for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
     for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
     this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this)); // 绘图循环
     this->loop_plot->start(); // 启动绘图循环
 #endif
 #ifdef CSV_LOGGER
     this->CSVInit(this->robot_name); // 初始化CSV记录
 #endif
 }
 
 // RL_Real类的析构函数，关闭所有循环
 RL_Real::~RL_Real()
 {
     this->loop_udpSend->shutdown(); // 关闭UDP发送循环
     this->loop_udpRecv->shutdown(); // 关闭UDP接收循环
     this->loop_keyboard->shutdown(); // 关闭键盘输入循环
     this->loop_control->shutdown(); // 关闭机器人控制循环
     this->loop_rl->shutdown(); // 关闭强化学习循环
 #ifdef PLOT
     this->loop_plot->shutdown(); // 关闭绘图循环
 #endif
     std::cout << LOGGER::INFO << "RL_Real exit" << std::endl; // 输出退出信息
 }
 
 // 获取机器人的状态
 void RL_Real::GetState(RobotState<double> *state)
 {
     this->unitree_udp.GetRecv(this->unitree_low_state); // 从UDP接收状态
     memcpy(&this->unitree_joy, this->unitree_low_state.wirelessRemote, 40); // 复制遥控器状态，40是遥控器状态的大小
 
     // 根据遥控器按钮更新控制状态
     if ((int)this->unitree_joy.btn.components.R2 == 1)
     {
         this->control.control_state = STATE_POS_GETUP; // 起身
     }
     else if ((int)this->unitree_joy.btn.components.R1 == 1)
     {
         this->control.control_state = STATE_RL_INIT; // RL初始化
     }
     else if ((int)this->unitree_joy.btn.components.L2 == 1)
     {
         this->control.control_state = STATE_POS_GETDOWN; // 躺下
     }
 
     // 根据框架设置IMU四元数
     if (this->params.framework == "isaacgym")
     {
         state->imu.quaternion[3] = this->unitree_low_state.imu.quaternion[0]; // w
         state->imu.quaternion[0] = this->unitree_low_state.imu.quaternion[1]; // x
         state->imu.quaternion[1] = this->unitree_low_state.imu.quaternion[2]; // y
         state->imu.quaternion[2] = this->unitree_low_state.imu.quaternion[3]; // z
     }
     else if (this->params.framework == "isaacsim")
     {
         state->imu.quaternion[0] = this->unitree_low_state.imu.quaternion[0]; // w
         state->imu.quaternion[1] = this->unitree_low_state.imu.quaternion[1]; // x
         state->imu.quaternion[2] = this->unitree_low_state.imu.quaternion[2]; // y
         state->imu.quaternion[3] = this->unitree_low_state.imu.quaternion[3]; // z
     }
 
     // 复制陀螺仪和加速度计数据
     for (int i = 0; i < 3; ++i)
     {
         state->imu.gyroscope[i] = this->unitree_low_state.imu.gyroscope[i];
     }
     for (int i = 0; i < this->params.num_of_dofs; ++i)
     {
         state->motor_state.q[i] = this->unitree_low_state.motorState[this->params.state_mapping[i]].q; // 关节位置
         state->motor_state.dq[i] = this->unitree_low_state.motorState[this->params.state_mapping[i]].dq; // 关节速度
         state->motor_state.tau_est[i] = this->unitree_low_state.motorState[this->params.state_mapping[i]].tauEst; // 估计的扭矩
     }
 }
 
 // 设置机器人的控制命令
 void RL_Real::SetCommand(const RobotCommand<double> *command)
 {
     for (int i = 0; i < this->params.num_of_dofs; ++i)
     {
         this->unitree_low_command.motorCmd[i].mode = 0x0A; // 设置控制模式
         this->unitree_low_command.motorCmd[i].q = command->motor_command.q[this->params.command_mapping[i]]; // 设置目标位置
         this->unitree_low_command.motorCmd[i].dq = command->motor_command.dq[this->params.command_mapping[i]]; // 设置目标速度
         this->unitree_low_command.motorCmd[i].Kp = command->motor_command.kp[this->params.command_mapping[i]]; // 设置位置控制增益
         this->unitree_low_command.motorCmd[i].Kd = command->motor_command.kd[this->params.command_mapping[i]]; // 设置速度控制增益
         this->unitree_low_command.motorCmd[i].tau = command->motor_command.tau[this->params.command_mapping[i]]; // 设置目标扭矩
     }
 
     // 执行保护措施
     this->unitree_safe.PowerProtect(this->unitree_low_command, this->unitree_low_state, 8);
     // this->unitree_safe.PositionProtect(this->unitree_low_command, this->unitree_low_state);
     this->unitree_udp.SetSend(this->unitree_low_command); // 发送控制命令
 }
 
 // 机器人控制主循环
 void RL_Real::RobotControl()
 {
     this->motiontime++; // 增加运动时间计数
 
     this->GetState(&this->robot_state); // 获取当前状态
     this->StateController(&this->robot_state, &this->robot_command); // 状态控制，从状态到控制
     this->SetCommand(&this->robot_command); // 设置控制命令，到下发控制
 }
 
 // 运行模型进行推理，然后这些模型会被送到rl——sdk中完成运动控制指令输出
 void RL_Real::RunModel()
 {
     if (this->running_state == STATE_RL_RUNNING) // 如果处于RL运行状态
     {
         // 更新观察数据
         this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0); // 角速度
         this->obs.commands = torch::tensor({{this->unitree_joy.ly, -this->unitree_joy.rx, -this->unitree_joy.lx}}); // 控制命令
         this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0); // 基础四元数
         this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0); // 自由度位置
         this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0); // 自由度速度
 
         // 前向推理
         this->obs.actions = this->Forward();
         this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau); // 计算输出，这些输出又会传入到rl_sdk中完成控制，通过kp增益来完成控制
 
         // 将输出推送到队列
         if (this->output_dof_pos.defined() && this->output_dof_pos.numel() > 0)
         {
             output_dof_pos_queue.push(this->output_dof_pos);
         }
         if (this->output_dof_vel.defined() && this->output_dof_vel.numel() > 0)
         {
             output_dof_vel_queue.push(this->output_dof_vel);
         }
         if (this->output_dof_tau.defined() && this->output_dof_tau.numel() > 0)
         {
             output_dof_tau_queue.push(this->output_dof_tau);
         }
 
         // 保护措施（可选）
         // this->TorqueProtect(this->output_dof_tau);
         // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);
 
 #ifdef CSV_LOGGER
         // 记录数据到CSV
         torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tau_est).unsqueeze(0);
         this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
 #endif
     }
 }
 
 // 前向推理函数
 torch::Tensor RL_Real::Forward()
 {
     torch::autograd::GradMode::set_enabled(false); // 禁用梯度计算，因为这里不需要计算梯度，只有在训练的时候需要计算梯度
 
     torch::Tensor clamped_obs = this->ComputeObservation(); // 计算观察数据，这里面数据是从imu这些拿到的
 
     torch::Tensor actions; // 存储动作
     if (!this->params.observations_history.empty()) // 如果需要有观察历史，这个一般告诉你需要获取vector中第多少位的数据
     {
         this->history_obs_buf.insert(clamped_obs); // 插入当前观察
         this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history); // 获取历史观察
         actions = this->model.forward({this->history_obs}).toTensor(); // 前向推理
     }
     else
     {
         actions = this->model.forward({clamped_obs}).toTensor(); // 前向推理
     }
 
     // 限制动作范围
     if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
     {
         return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper); // 返回限制后的动作
     }
     else
     {
         return actions; // 返回原始动作
     }
 }
 
 // 绘图函数
 void RL_Real::Plot()
 {
     this->plot_t.erase(this->plot_t.begin()); // 移除最旧的时间点
     this->plot_t.push_back(this->motiontime); // 添加当前时间点
     plt::cla(); // 清除当前图形
     plt::clf(); // 清除当前图形窗口
     for (int i = 0; i < this->params.num_of_dofs; ++i) // 遍历所有自由度
     {
         this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin()); // 移除最旧的关节位置
         this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin()); // 移除最旧的目标关节位置
         this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motorState[i].q); // 添加当前关节位置
         this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motorCmd[i].q); // 添加目标关节位置
         plt::subplot(4, 3, i + 1); // 创建子图
         plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r"); // 绘制实际关节位置
         plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b"); // 绘制目标关节位置
         plt::xlim(this->plot_t.front(), this->plot_t.back()); // 设置x轴范围
     }
     // plt::legend(); // 显示图例（可选）
     plt::pause(0.0001); // 暂停以更新图形
 }
 
 // 信号处理函数，用于捕获中断信号
 void signalHandler(int signum)
 {
     exit(0); // 退出程序
 }
 
 // 主函数
 int main(int argc, char **argv)
 {
     signal(SIGINT, signalHandler); // 注册信号处理函数
 
     RL_Real rl_sar; // 创建RL_Real对象
 
     while (1) // 持续运行
     {
         sleep(10); // 每10秒循环一次
     }
 
     return 0; // 返回0，表示正常结束
 }
 