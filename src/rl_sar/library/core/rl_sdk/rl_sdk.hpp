/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

 #ifndef RL_SDK_HPP
 #define RL_SDK_HPP
 
 #include <torch/script.h>
 #include <iostream>
 #include <string>
 #include <unistd.h>
 #include <tbb/concurrent_queue.h>
 
 #include <yaml-cpp/yaml.h>
 
 namespace LOGGER
 {
     // 日志级别常量定义
     const char *const INFO    = "\033[0;37m[INFO]\033[0m ";    // 信息日志
     const char *const WARNING = "\033[0;33m[WARNING]\033[0m "; // 警告日志
     const char *const ERROR   = "\033[0;31m[ERROR]\033[0m ";   // 错误日志
     const char *const DEBUG   = "\033[0;32m[DEBUG]\033[0m ";   // 调试日志
 }
 
 // RobotCommand结构体用于表示机器人控制命令
 template <typename T>
 struct RobotCommand
 {
     struct MotorCommand
     {
         std::vector<int> mode = std::vector<int>(32, 0); // 控制模式，默认值为0
         std::vector<T> q = std::vector<T>(32, 0.0);      // 关节位置
         std::vector<T> dq = std::vector<T>(32, 0.0);     // 关节速度
         std::vector<T> tau = std::vector<T>(32, 0.0);    // 关节扭矩
         std::vector<T> kp = std::vector<T>(32, 0.0);     // 位置控制增益
         std::vector<T> kd = std::vector<T>(32, 0.0);     // 速度控制增益
     } motor_command; // 电机控制命令
 };
 
 // RobotState结构体用于表示机器人的状态
 template <typename T>
 struct RobotState
 {
     struct IMU
     {
         std::vector<T> quaternion = {1.0, 0.0, 0.0, 0.0}; // 四元数（w, x, y, z）
         std::vector<T> gyroscope = {0.0, 0.0, 0.0};       // 陀螺仪数据
         std::vector<T> accelerometer = {0.0, 0.0, 0.0};   // 加速度计数据
     } imu; // IMU传感器状态
 
     struct MotorState
     {
         std::vector<T> q = std::vector<T>(32, 0.0);      // 关节位置
         std::vector<T> dq = std::vector<T>(32, 0.0);     // 关节速度
         std::vector<T> ddq = std::vector<T>(32, 0.0);    // 关节加速度
         std::vector<T> tau_est = std::vector<T>(32, 0.0); // 估计的关节扭矩
         std::vector<T> cur = std::vector<T>(32, 0.0);     // 电流
     } motor_state; // 电机状态
 };
 
 // 状态枚举定义
 enum STATE
 {
     STATE_WAITING = 0,      // 等待状态
     STATE_POS_GETUP,        // 起身状态
     STATE_RL_INIT,          // RL初始化状态
     STATE_RL_RUNNING,       // RL运行状态
     STATE_POS_GETDOWN,      // 躺下状态
     STATE_RESET_SIMULATION, // 重置仿真状态
     STATE_TOGGLE_SIMULATION, // 切换仿真状态
 };
 
 // Control结构体用于表示控制信息
 struct Control
 {
     STATE control_state; // 当前控制状态
     double x = 0.0;      // x坐标
     double y = 0.0;      // y坐标
     double yaw = 0.0;    // 偏航角
     double wheel = 0.0;  // 轮子状态
 };
 
 // ModelParams结构体用于表示模型参数
 struct ModelParams
 {
     std::string model_name; // 模型名称
     std::string framework;   // 框架名称
     double dt;               // 时间步长
     int decimation;          // 抽样率
     int num_observations;    // 观察数量
     std::vector<std::string> observations; // 观察参数列表
     std::vector<int> observations_history; // 观察历史
     double damping;          // 阻尼系数
     double stiffness;        // 刚度系数
     torch::Tensor action_scale; // 动作缩放因子
     std::vector<int> wheel_indices; // 轮子索引
     int num_of_dofs;        // 自由度数量
     double lin_vel_scale;   // 线速度缩放因子
     double ang_vel_scale;   // 角速度缩放因子
     double dof_pos_scale;   // 自由度位置缩放因子
     double dof_vel_scale;   // 自由度速度缩放因子
     double clip_obs;        // 观察限制
     torch::Tensor clip_actions_upper; // 动作上限
     torch::Tensor clip_actions_lower; // 动作下限
     torch::Tensor torque_limits; // 扭矩限制
     torch::Tensor rl_kd;    // RL速度控制增益
     torch::Tensor rl_kp;    // RL位置控制增益
     torch::Tensor fixed_kp;  // 固定位置控制增益
     torch::Tensor fixed_kd;  // 固定速度控制增益
     torch::Tensor commands_scale; // 控制命令缩放因子
     torch::Tensor default_dof_pos; // 默认自由度位置
     std::vector<std::string> joint_controller_names; // 关节控制器名称
     std::vector<int> command_mapping; // 命令映射
     std::vector<int> state_mapping; // 状态映射
 };
 
 // Observations结构体用于表示观察数据
 struct Observations
 {
     torch::Tensor lin_vel;    // 线速度
     torch::Tensor ang_vel;    // 角速度
     torch::Tensor gravity_vec; // 重力向量
     torch::Tensor commands;    // 控制命令
     torch::Tensor base_quat;   // 基础四元数
     torch::Tensor dof_pos;     // 自由度位置
     torch::Tensor dof_vel;     // 自由度速度
     torch::Tensor actions;      // 动作
 };
 
 class RL
 {
 public:
     RL() {};
     ~RL() {};
 
     ModelParams params;
     Observations obs;
 
     RobotState<double> robot_state;
     RobotCommand<double> robot_command;
     tbb::concurrent_queue<torch::Tensor> output_dof_pos_queue;
     tbb::concurrent_queue<torch::Tensor> output_dof_vel_queue;
     tbb::concurrent_queue<torch::Tensor> output_dof_tau_queue;
 
     // init
     void InitObservations();
     void InitOutputs();
     void InitControl();
 
     // rl functions
     virtual torch::Tensor Forward() = 0;
     torch::Tensor ComputeObservation();
     virtual void GetState(RobotState<double> *state) = 0;
     virtual void SetCommand(const RobotCommand<double> *command) = 0;
     void StateController(const RobotState<double> *state, RobotCommand<double> *command);
     void ComputeOutput(const torch::Tensor &actions, torch::Tensor &output_dof_pos, torch::Tensor &output_dof_vel, torch::Tensor &output_dof_tau);
     torch::Tensor QuatRotateInverse(torch::Tensor q, torch::Tensor v, const std::string &framework);
 
     // yaml params
     void ReadYaml(std::string robot_path);
 
     // csv logger
     std::string csv_filename;
     void CSVInit(std::string robot_name);
     void CSVLogger(torch::Tensor torque, torch::Tensor tau_est, torch::Tensor joint_pos, torch::Tensor joint_pos_target, torch::Tensor joint_vel);
 
     // control
     Control control;
     void KeyboardInterface();
 
     // others
     std::string robot_name, config_name;
     STATE running_state = STATE_RL_RUNNING; // default running_state set to STATE_RL_RUNNING
     bool simulation_running = false;
 
     // protect func
     void TorqueProtect(torch::Tensor origin_output_dof_tau);
     void AttitudeProtect(const std::vector<double> &quaternion, float pitch_threshold, float roll_threshold);
 
 protected:
     // rl module
     torch::jit::script::Module model;
     // output buffer
     torch::Tensor output_dof_tau;
     torch::Tensor output_dof_pos;
     torch::Tensor output_dof_vel;
 };
 
 template <typename T>
 T clamp(T value, T min, T max)
 {
     if (value < min) return min;
     if (value > max) return max;
     return value;
 }
 
 #endif // RL_SDK_HPP
 