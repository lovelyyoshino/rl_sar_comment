/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

 #include "rl_sdk.hpp"

 /* You may need to override this Forward() function
 torch::Tensor RL_XXX::Forward()
 {
     torch::autograd::GradMode::set_enabled(false);
     torch::Tensor clamped_obs = this->ComputeObservation();
     torch::Tensor actions = this->model.forward({clamped_obs}).toTensor();
     torch::Tensor clamped_actions = torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
     return clamped_actions;
 }
 */
 
 // 计算观察数据
 torch::Tensor RL::ComputeObservation()
 {
     std::vector<torch::Tensor> obs_list; // 用于存储观察数据的列表
 
     // 遍历所有观察参数
     for (const std::string &observation : this->params.observations)
     {
         if (observation == "lin_vel") // 线速度
         {
             obs_list.push_back(this->obs.lin_vel * this->params.lin_vel_scale); // 将线速度缩放后添加到列表,输入的时候就是Tensor
         }
         /*
             QuatRotateInverse函数的第一个参数是表示机器人方向的四元数，第二个参数是在世界坐标系中的向量。该函数输出第二个参数在机体坐标系中的值。
             在IsaacGym中，角速度的坐标系是世界坐标系。在训练过程中，观察中的角速度使用QuatRotateInverse将坐标系转换为机体坐标系。
             在Gazebo中，角速度的坐标系也是世界坐标系，因此需要使用QuatRotateInverse将坐标系转换为机体坐标系。
             在一些真实机器人（如Unitree）中，如果角速度的坐标系已经在机体坐标系中，则不需要进行转换。
             忘记进行转换或多次进行转换可能会导致控制器在旋转达到180度时崩溃。
         */
         else if (observation == "ang_vel_body") // 机体坐标系下的角速度
         {
             obs_list.push_back(this->obs.ang_vel * this->params.ang_vel_scale); // 将角速度缩放后添加到列表
         }
         else if (observation == "ang_vel_world") // 世界坐标系下的角速度
         {
             obs_list.push_back(this->QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel, this->params.framework) * this->params.ang_vel_scale); // 转换并缩放后添加到列表
         }
         else if (observation == "gravity_vec") // 重力向量
         {
             obs_list.push_back(this->QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec, this->params.framework)); // 转换重力向量并添加到列表
         }
         else if (observation == "commands") // 控制命令
         {
             obs_list.push_back(this->obs.commands * this->params.commands_scale); // 将控制命令缩放后添加到列表
         }
         else if (observation == "dof_pos") // 自由度位置
         {
             torch::Tensor dof_pos_rel = this->obs.dof_pos - this->params.default_dof_pos; // 计算相对位置
             for (int i : this->params.wheel_indices) // 对于轮子索引
             {
                 dof_pos_rel[0][i] = 0.0; // 将轮子位置设为0
             }
             obs_list.push_back(dof_pos_rel * this->params.dof_pos_scale); // 将相对位置缩放后添加到列表
         }
         else if (observation == "dof_vel") // 自由度速度
         {
             obs_list.push_back(this->obs.dof_vel * this->params.dof_vel_scale); // 将速度缩放后添加到列表
         }
         else if (observation == "actions") // 动作
         {
             obs_list.push_back(this->obs.actions); // 添加动作到列表
         }
     }
 
     // 将所有观察数据连接成一个Tensor，1表示列，0表示行，意思是将所有列连接成一个Tensor
     torch::Tensor obs = torch::cat(obs_list, 1);
     // 将观察数据限制在指定范围内，torch::clamp()函数将输入的值限制在指定范围内，-this->params.clip_obs和this->params.clip_obs是限制的范围
     torch::Tensor clamped_obs = torch::clamp(obs, -this->params.clip_obs, this->params.clip_obs);
     return clamped_obs; // 返回限制后的观察数据
 }
 
 void RL::InitObservations()
 {
     // 初始化观察数据
     this->obs.lin_vel = torch::tensor({{0.0, 0.0, 0.0}}); // 线速度初始化为0
     this->obs.ang_vel = torch::tensor({{0.0, 0.0, 0.0}}); // 角速度初始化为0
     this->obs.gravity_vec = torch::tensor({{0.0, 0.0, -1.0}}); // 重力向量初始化为负z方向
     this->obs.commands = torch::tensor({{0.0, 0.0, 0.0}}); // 控制命令初始化为0
     this->obs.base_quat = torch::tensor({{0.0, 0.0, 0.0, 1.0}}); // 基础四元数初始化为单位四元数
     this->obs.dof_pos = this->params.default_dof_pos; // 自由度位置初始化为默认位置
     this->obs.dof_vel = torch::zeros({1, this->params.num_of_dofs}); // 自由度速度初始化为0，一行，num_of_dofs列
     this->obs.actions = torch::zeros({1, this->params.num_of_dofs}); // 动作初始化为0，一行，num_of_dofs列
 }
 
 void RL::InitOutputs()
 {
     // 初始化输出数据
     this->output_dof_tau = torch::zeros({1, this->params.num_of_dofs}); // 自由度扭矩初始化为0
     this->output_dof_pos = this->params.default_dof_pos; // 自由度位置初始化为默认位置
     this->output_dof_vel = torch::zeros({1, this->params.num_of_dofs}); // 自由度速度初始化为0
 }
 
 void RL::InitControl()
 {
     // 初始化控制状态
     this->control.control_state = STATE_WAITING; // 控制状态初始化为等待
     this->control.x = 0.0; // x坐标初始化为0
     this->control.y = 0.0; // y坐标初始化为0
     this->control.yaw = 0.0; // 偏航角初始化为0
 }
 
 //这里output_dof_pos, output_dof_vel, output_dof_tau就是this->output_dof_pos, this->output_dof_vel, this->output_dof_tau，是从rl_real_a1.cpp中传入的.
 void RL::ComputeOutput(const torch::Tensor &actions, torch::Tensor &output_dof_pos, torch::Tensor &output_dof_vel, torch::Tensor &output_dof_tau)
 {
     // 将动作缩放
     torch::Tensor actions_scaled = actions * this->params.action_scale;
     torch::Tensor pos_actions_scaled = actions_scaled.clone(); // 位置动作缩放
     torch::Tensor vel_actions_scaled = torch::zeros_like(actions); // 速度动作初始化为0
     for (int i : this->params.wheel_indices) // 对于轮子索引，这个和上面对应对应，将轮子索引的数据设置为位置为0，速度为缩放后的速度
     {
         pos_actions_scaled[0][i] = 0.0; // 将轮子位置动作设为0
         vel_actions_scaled[0][i] = actions_scaled[0][i]; // 将轮子速度动作设为缩放后的速度
     }
     // 计算所有动作的总和，这个数值和actions_scaled的数值是一样的。
     torch::Tensor all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
     output_dof_pos = pos_actions_scaled + this->params.default_dof_pos; // 计算输出位置，这里考虑到默认的pos位置
     output_dof_vel = vel_actions_scaled; // 计算输出速度
     // 计算输出扭矩，这里kp和kd是RL控制器，all_actions_scaled是位置和速度的和，this->params.default_dof_pos是默认的位置，this->obs.dof_pos是当前的位置，this->obs.dof_vel是当前的速度
     output_dof_tau = this->params.rl_kp * (all_actions_scaled + this->params.default_dof_pos - this->obs.dof_pos) - this->params.rl_kd * this->obs.dof_vel;
     // 限制扭矩在指定范围内
     output_dof_tau = torch::clamp(output_dof_tau, -(this->params.torque_limits), this->params.torque_limits);
 }
 
 torch::Tensor RL::QuatRotateInverse(torch::Tensor q, torch::Tensor v, const std::string &framework)
 {
     torch::Tensor q_w; // 四元数的标量部分
     torch::Tensor q_vec; // 四元数的向量部分
     if (framework == "isaacsim") // 如果框架是isaacsim
     {
         q_w = q.index({torch::indexing::Slice(), 0}); // 获取四元数的标量部分，0表示第一列
         q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)}); // 获取四元数的向量部分
     }
     else if (framework == "isaacgym") // 如果框架是isaacgym
     {
         q_w = q.index({torch::indexing::Slice(), 3}); // 获取四元数的标量部分
         q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); // 获取四元数的向量部分
     }
     c10::IntArrayRef shape = q.sizes(); // 获取四元数的形状
 
     // 计算旋转后的向量
     torch::Tensor a = v * (2.0 * torch::pow(q_w, 2) - 1.0).unsqueeze(-1);
     torch::Tensor b = torch::cross(q_vec, v, -1) * q_w.unsqueeze(-1) * 2.0;
     torch::Tensor c = q_vec * torch::bmm(q_vec.view({shape[0], 1, 3}), v.view({shape[0], 3, 1})).squeeze(-1) * 2.0;
     return a - b + c; // 返回旋转后的向量
 }
 
 //也是通过线程控制的
 void RL::StateController(const RobotState<double> *state, RobotCommand<double> *command)
 {
     static RobotState<double> start_state; // 起始状态
     static RobotState<double> now_state; // 当前状态
     static float getup_percent = 0.0; // 起身百分比
     static float getdown_percent = 0.0; // 躺下百分比
 
     // 等待状态
     if (this->running_state == STATE_WAITING)
     {
         for (int i = 0; i < this->params.num_of_dofs; ++i)//对应的关节角度
         {
             command->motor_command.q[i] = state->motor_state.q[i]; // 将当前状态的关节位置赋值给命令，最多32个字节
         }
         if (this->control.control_state == STATE_POS_GETUP) // 如果控制状态是起身
         {
             this->control.control_state = STATE_WAITING; // 切换控制状态为等待
             getup_percent = 0.0; // 重置起身百分比
             for (int i = 0; i < this->params.num_of_dofs; ++i)
             {
                 now_state.motor_state.q[i] = state->motor_state.q[i]; // 更新当前状态
                 start_state.motor_state.q[i] = now_state.motor_state.q[i]; // 设置起始状态
             }
             this->running_state = STATE_POS_GETUP; // 切换运行状态为起身
             std::cout << std::endl << LOGGER::INFO << "Switching to STATE_POS_GETUP" << std::endl;
         }
     }
     // 起身（位置控制）
     else if (this->running_state == STATE_POS_GETUP)
     {
         if (getup_percent < 1.0) // 如果起身百分比小于1
         {
             getup_percent += 1 / 500.0; // 增加起身百分比
             getup_percent = getup_percent > 1.0 ? 1.0 : getup_percent; // 限制最大值为1
             for (int i = 0; i < this->params.num_of_dofs; ++i)
             {
                 command->motor_command.q[i] = (1 - getup_percent) * now_state.motor_state.q[i] + getup_percent * this->params.default_dof_pos[0][i].item<double>(); // 计算插值位置
                 command->motor_command.dq[i] = 0; // 速度设为0
                 command->motor_command.kp[i] = this->params.fixed_kp[0][i].item<double>(); // 设置位置控制增益
                 command->motor_command.kd[i] = this->params.fixed_kd[0][i].item<double>(); // 设置速度控制增益
                 command->motor_command.tau[i] = 0; // 扭矩设为0
             }
             std::cout << "\r" << std::flush << LOGGER::INFO << "Getting up " << std::fixed << std::setprecision(2) << getup_percent * 100.0 << std::flush; // 输出起身进度
         }
         else // 如果起身完成
         {
             if (this->control.control_state == STATE_RL_INIT) // 如果控制状态是RL初始化
             {
                 this->control.control_state = STATE_WAITING; // 切换控制状态为等待
                 this->running_state = STATE_RL_INIT; // 切换运行状态为RL初始化
                 std::cout << std::endl << LOGGER::INFO << "Switching to STATE_RL_INIT" << std::endl;
             }
             else if (this->control.control_state == STATE_POS_GETDOWN) // 如果控制状态是躺下
             {
                 this->control.control_state = STATE_WAITING; // 切换控制状态为等待
                 getdown_percent = 0.0; // 重置躺下百分比
                 for (int i = 0; i < this->params.num_of_dofs; ++i)
                 {
                     now_state.motor_state.q[i] = state->motor_state.q[i]; // 更新当前状态
                 }
                 this->running_state = STATE_POS_GETDOWN; // 切换运行状态为躺下
                 std::cout << std::endl << LOGGER::INFO << "Switching to STATE_POS_GETDOWN" << std::endl;
             }
         }
     }
     // 初始化观察并开始RL循环
     else if (this->running_state == STATE_RL_INIT)
     {
         if (getup_percent == 1) // 如果起身完成
         {
             this->InitObservations(); // 初始化观察
             this->InitOutputs(); // 初始化输出
             this->InitControl(); // 初始化控制
             this->running_state = STATE_RL_RUNNING; // 切换运行状态为RL运行
             std::cout << std::endl << LOGGER::INFO << "Switching to STATE_RL_RUNNING" << std::endl;
         }
     }
     // RL循环
     else if (this->running_state == STATE_RL_RUNNING)
     {
         std::cout << "\r" << std::flush << LOGGER::INFO << "RL Controller x:" << this->control.x << " y:" << this->control.y << " yaw:" << this->control.yaw << std::flush;
 
         torch::Tensor _output_dof_pos, _output_dof_vel; // 输出位置和速度
         // 尝试从队列中获取输出位置和速度，这个会从例如rl_real_a1.cpp中传入，然后传入到这个函数中取出rl算出来的结果。因为output_dof_pos_queue是public的，然后RL_Real是RL的子类，所以可以访问到。
         if (this->output_dof_pos_queue.try_pop(_output_dof_pos) && this->output_dof_vel_queue.try_pop(_output_dof_vel))
         {
             for (int i = 0; i < this->params.num_of_dofs; ++i)
             {
                 if (_output_dof_pos.defined() && _output_dof_pos.numel() > 0) // 如果输出位置有效
                 {
                     command->motor_command.q[i] = this->output_dof_pos[0][i].item<double>(); // 设置关节位置
                 }
                 if (_output_dof_vel.defined() && _output_dof_vel.numel() > 0) // 如果输出速度有效
                 {
                     command->motor_command.dq[i] = this->output_dof_vel[0][i].item<double>(); // 设置关节速度
                 }
                 command->motor_command.kp[i] = this->params.rl_kp[0][i].item<double>(); // 设置位置控制增益
                 command->motor_command.kd[i] = this->params.rl_kd[0][i].item<double>(); // 设置速度控制增益
                 command->motor_command.tau[i] = 0; // 扭矩设为0
             }
         }
         if (this->control.control_state == STATE_POS_GETDOWN) // 如果控制状态是躺下
         {
             this->control.control_state = STATE_WAITING; // 切换控制状态为等待
             getdown_percent = 0.0; // 重置躺下百分比
             for (int i = 0; i < this->params.num_of_dofs; ++i)
             {
                 now_state.motor_state.q[i] = state->motor_state.q[i]; // 更新当前状态，因为需要使用当前姿态和默认的躺下姿态做插值
             }
             this->running_state = STATE_POS_GETDOWN; // 切换运行状态为躺下
             std::cout << std::endl << LOGGER::INFO << "Switching to STATE_POS_GETDOWN" << std::endl;
         }
         else if (this->control.control_state == STATE_POS_GETUP) // 如果控制状态是起身
         {
             this->control.control_state = STATE_WAITING; // 切换控制状态为等待
             getup_percent = 0.0; // 重置起身百分比
             for (int i = 0; i < this->params.num_of_dofs; ++i)
             {
                 now_state.motor_state.q[i] = state->motor_state.q[i]; // 更新当前状态，因为需要使用当前姿态和默认的站起姿态做插值
             }
             this->running_state = STATE_POS_GETUP; // 切换运行状态为起身
             std::cout << std::endl << LOGGER::INFO << "Switching to STATE_POS_GETUP" << std::endl;
         }
     }
     // 躺下（位置控制）
     else if (this->running_state == STATE_POS_GETDOWN)
     {
         if (getdown_percent < 1.0) // 如果躺下百分比小于1
         {
             getdown_percent += 1 / 500.0; // 增加躺下百分比
             getdown_percent = getdown_percent > 1.0 ? 1.0 : getdown_percent; // 限制最大值为1
             for (int i = 0; i < this->params.num_of_dofs; ++i)
             {
                 command->motor_command.q[i] = (1 - getdown_percent) * now_state.motor_state.q[i] + getdown_percent * start_state.motor_state.q[i]; // 计算插值位置
                 command->motor_command.dq[i] = 0; // 速度设为0
                 command->motor_command.kp[i] = this->params.fixed_kp[0][i].item<double>(); // 设置位置控制增益
                 command->motor_command.kd[i] = this->params.fixed_kd[0][i].item<double>(); // 设置速度控制增益
                 command->motor_command.tau[i] = 0; // 扭矩设为0
             }
             std::cout << "\r" << std::flush << LOGGER::INFO << "Getting down " << std::fixed << std::setprecision(2) << getdown_percent * 100.0 << std::flush; // 输出躺下进度
         }
         if (getdown_percent == 1) // 如果躺下完成
         {
             this->InitObservations(); // 初始化观察
             this->InitOutputs(); // 初始化输出
             this->InitControl(); // 初始化控制
             this->running_state = STATE_WAITING; // 切换运行状态为等待
             std::cout << std::endl << LOGGER::INFO << "Switching to STATE_WAITING" << std::endl;
         }
     }
 }
 
 /*
 检查扭矩是否超出范围，如果超出范围，则记录超出范围的扭矩索引和值，然后输出警告信息。
 */
 void RL::TorqueProtect(torch::Tensor origin_output_dof_tau)
 {
     std::vector<int> out_of_range_indices; // 存储超出范围的扭矩索引
     std::vector<double> out_of_range_values; // 存储超出范围的扭矩值
     for (int i = 0; i < origin_output_dof_tau.size(1); ++i) // 遍历所有关节的扭矩
     {
         double torque_value = origin_output_dof_tau[0][i].item<double>(); // 获取当前扭矩值
         double limit_lower = -this->params.torque_limits[0][i].item<double>(); // 获取下限
         double limit_upper = this->params.torque_limits[0][i].item<double>(); // 获取上限
 
         // 检查扭矩是否超出范围
         if (torque_value < limit_lower || torque_value > limit_upper)
         {
             out_of_range_indices.push_back(i); // 记录超出范围的索引
             out_of_range_values.push_back(torque_value); // 记录超出范围的值
         }
     }
     // 如果有超出范围的扭矩
     if (!out_of_range_indices.empty())
     {
         for (int i = 0; i < out_of_range_indices.size(); ++i) // 遍历所有超出范围的扭矩
         {
             int index = out_of_range_indices[i]; // 获取索引
             double value = out_of_range_values[i]; // 获取扭矩值
             double limit_lower = -this->params.torque_limits[0][index].item<double>(); // 获取下限
             double limit_upper = this->params.torque_limits[0][index].item<double>(); // 获取上限
 
             // 输出警告信息
             std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
         }
         // 这里可以添加保护措施，例如切换控制状态
         // this->control.control_state = STATE_POS_GETDOWN;
         // std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
     }
 }
 
 void RL::AttitudeProtect(const std::vector<double> &quaternion, float pitch_threshold, float roll_threshold)
 {
     float rad2deg = 57.2958; // 弧度转度数的转换因子
     float w, x, y, z;
 
     // 根据框架选择四元数的分量
     if (this->params.framework == "isaacgym")
     {
         w = quaternion[3];
         x = quaternion[0];
         y = quaternion[1];
         z = quaternion[2];
     }
     else if (this->params.framework == "isaacsim")
     {
         w = quaternion[0];
         x = quaternion[1];
         y = quaternion[2];
         z = quaternion[3];
     }
 
     // 计算滚转角（绕X轴旋转）
     float sinr_cosp = 2 * (w * x + y * z);
     float cosr_cosp = 1 - 2 * (x * x + y * y);
     float roll = std::atan2(sinr_cosp, cosr_cosp) * rad2deg;
 
     // 计算俯仰角（绕Y轴旋转）
     float sinp = 2 * (w * y - z * x);
     float pitch;
     if (std::fabs(sinp) >= 1) // 限制俯仰角的范围
     {
         pitch = std::copysign(90.0, sinp); // 限制到90度
     }
     else
     {
         pitch = std::asin(sinp) * rad2deg; // 计算俯仰角
     }
 
     // 检查滚转角是否超过阈值
     if (std::fabs(roll) > roll_threshold)
     {
         // this->control.control_state = STATE_POS_GETDOWN; // 可以添加保护措施
         std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
     }
     // 检查俯仰角是否超过阈值
     if (std::fabs(pitch) > pitch_threshold)
     {
         // this->control.control_state = STATE_POS_GETDOWN; // 可以添加保护措施
         std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
     }
 }
 
 // 检测键盘输入的函数
 #include <termios.h>
 #include <sys/ioctl.h>
 static bool kbhit()
 {
     termios term;
     tcgetattr(0, &term); // 获取当前终端设置
 
     termios term2 = term;
     term2.c_lflag &= ~ICANON; // 设置为非规范模式
     tcsetattr(0, TCSANOW, &term2); // 应用新的设置
 
     int byteswaiting;
     ioctl(0, FIONREAD, &byteswaiting); // 检查输入缓冲区的字节数
 
     tcsetattr(0, TCSANOW, &term); // 恢复原来的设置
 
     return byteswaiting > 0; // 返回是否有输入
 }
 
 void RL::KeyboardInterface()
 {
     if (kbhit()) // 如果有键盘输入
     {
         int c = fgetc(stdin); // 获取输入字符
         switch (c)
         {
         case '0': // 按下'0'键
             this->control.control_state = STATE_POS_GETUP; // 切换到起身状态
             break;
         case 'p': // 按下'p'键
             this->control.control_state = STATE_RL_INIT; // 切换到RL初始化状态
             break;
         case '1': // 按下'1'键
             this->control.control_state = STATE_POS_GETDOWN; // 切换到躺下状态
             break;
         case 'q': // 按下'q'键
             break; // 不做任何操作
         case 'w': // 按下'w'键
             this->control.x += 0.1; // 增加x坐标
             break;
         case 's': // 按下's'键
             this->control.x -= 0.1; // 减少x坐标
             break;
         case 'a': // 按下'a'键
             this->control.yaw += 0.1; // 增加偏航角
             break;
         case 'd': // 按下'd'键
             this->control.yaw -= 0.1; // 减少偏航角
             break;
         case 'i': // 按下'i'键
             break; // 不做任何操作
         case 'k': // 按下'k'键
             break; // 不做任何操作
         case 'j': // 按下'j'键
             this->control.y += 0.1; // 增加y坐标
             break;
         case 'l': // 按下'l'键
             this->control.y -= 0.1; // 减少y坐标
             break;
         case ' ': // 按下空格键
             this->control.x = 0; // 重置x坐标
             this->control.y = 0; // 重置y坐标
             this->control.yaw = 0; // 重置偏航角
             break;
         case 'r': // 按下'r'键
             this->control.control_state = STATE_RESET_SIMULATION; // 切换到重置仿真状态
             break;
         case '\n': // 按下回车键
             this->control.control_state = STATE_TOGGLE_SIMULATION; // 切换仿真状态
             break;
         default:
             break; // 不做任何操作
         }
     }
 }
 
 // 从YAML节点读取向量的模板函数
 template <typename T>
 std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
 {
     std::vector<T> values; // 存储读取的值
     for (const auto &val : node) // 遍历节点
     {
         values.push_back(val.as<T>()); // 将值添加到向量中
     }
     return values; // 返回读取的向量
 }
 
 // 从YAML文件读取机器人参数
 void RL::ReadYaml(std::string robot_path)
 {
     // 配置文件路径
     std::string config_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + robot_path + "/config.yaml";
     YAML::Node config; // YAML节点
     try
     {
         config = YAML::LoadFile(config_path)[robot_path]; // 加载YAML文件
     }
     catch (YAML::BadFile &e) // 捕获文件错误
     {
         std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl; // 输出错误信息
         return;
     }
 
     // 读取配置参数
     this->params.model_name = config["model_name"].as<std::string>();
     this->params.framework = config["framework"].as<std::string>();
     this->params.dt = config["dt"].as<double>();
     this->params.decimation = config["decimation"].as<int>();
     this->params.num_observations = config["num_observations"].as<int>();
     this->params.observations = ReadVectorFromYaml<std::string>(config["observations"]); // 读取观察参数
     if (config["observations_history"].IsNull())
     {
         this->params.observations_history = {}; // 如果为空则初始化为空
     }
     else
     {
         this->params.observations_history = ReadVectorFromYaml<int>(config["observations_history"]); // 读取观察历史
     }
     this->params.clip_obs = config["clip_obs"].as<double>(); // 读取观察限制
     if (config["clip_actions_lower"].IsNull() && config["clip_actions_upper"].IsNull())
     {
         this->params.clip_actions_upper = torch::tensor({}).view({1, -1}); // 初始化动作上限
         this->params.clip_actions_lower = torch::tensor({}).view({1, -1}); // 初始化动作下限
     }
     else
     {
         this->params.clip_actions_upper = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_upper"])).view({1, -1}); // 读取动作上限
         this->params.clip_actions_lower = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_lower"])).view({1, -1}); // 读取动作下限
     }
     this->params.action_scale = torch::tensor(ReadVectorFromYaml<double>(config["action_scale"])).view({1, -1}); // 读取动作缩放因子
     this->params.wheel_indices = ReadVectorFromYaml<int>(config["wheel_indices"]); // 读取轮子索引
     this->params.num_of_dofs = config["num_of_dofs"].as<int>(); // 读取自由度数量
     this->params.lin_vel_scale = config["lin_vel_scale"].as<double>(); // 读取线速度缩放因子
     this->params.ang_vel_scale = config["ang_vel_scale"].as<double>(); // 读取角速度缩放因子
     this->params.dof_pos_scale = config["dof_pos_scale"].as<double>(); // 读取自由度位置缩放因子
     this->params.dof_vel_scale = config["dof_vel_scale"].as<double>(); // 读取自由度速度缩放因子
     this->params.commands_scale = torch::tensor(ReadVectorFromYaml<double>(config["commands_scale"])).view({1, -1}); // 读取控制命令缩放因子
     this->params.rl_kp = torch::tensor(ReadVectorFromYaml<double>(config["rl_kp"])).view({1, -1}); // 读取RL位置控制增益
     this->params.rl_kd = torch::tensor(ReadVectorFromYaml<double>(config["rl_kd"])).view({1, -1}); // 读取RL速度控制增益
     this->params.fixed_kp = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kp"])).view({1, -1}); // 读取固定位置控制增益
     this->params.fixed_kd = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kd"])).view({1, -1}); // 读取固定速度控制增益
     this->params.torque_limits = torch::tensor(ReadVectorFromYaml<double>(config["torque_limits"])).view({1, -1}); // 读取扭矩限制
     this->params.default_dof_pos = torch::tensor(ReadVectorFromYaml<double>(config["default_dof_pos"])).view({1, -1}); // 读取默认自由度位置
     this->params.joint_controller_names = ReadVectorFromYaml<std::string>(config["joint_controller_names"]); // 读取关节控制器名称
     this->params.command_mapping = ReadVectorFromYaml<int>(config["command_mapping"]); // 读取命令映射
     this->params.state_mapping = ReadVectorFromYaml<int>(config["state_mapping"]); // 读取状态映射
 }
 
 // 初始化CSV文件
 void RL::CSVInit(std::string robot_name)
 {
     csv_filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + robot_name + "/motor"; // CSV文件路径
 
     // 如果需要时间戳，可以取消注释以下代码
     // auto now = std::chrono::system_clock::now();
     // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
     // std::stringstream ss;
     // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
     // std::string timestamp = ss.str();
     // csv_filename += "_" + timestamp;
 
     csv_filename += ".csv"; // 添加文件扩展名
     std::ofstream file(csv_filename.c_str()); // 打开CSV文件
 
     // 写入CSV文件的表头
     // 写入CSV文件的表头
     for(int i = 0; i < 12; ++i) { file << "tau_cal_" << i << ","; } // 写入计算的扭矩
     for(int i = 0; i < 12; ++i) { file << "tau_est_" << i << ","; } // 写入估计的扭矩
     for(int i = 0; i < 12; ++i) { file << "joint_pos_" << i << ","; } // 写入关节位置
     for(int i = 0; i < 12; ++i) { file << "joint_pos_target_" << i << ","; } // 写入目标关节位置
     for(int i = 0; i < 12; ++i) { file << "joint_vel_" << i << ","; } // 写入关节速度
 
     file << std::endl; // 换行
 
     file.close(); // 关闭文件
 }
 
 // 记录数据到CSV文件
 void RL::CSVLogger(torch::Tensor torque, torch::Tensor tau_est, torch::Tensor joint_pos, torch::Tensor joint_pos_target, torch::Tensor joint_vel)
 {
     std::ofstream file(csv_filename.c_str(), std::ios_base::app); // 以追加模式打开CSV文件
 
     // 写入扭矩数据
     for(int i = 0; i < 12; ++i) { file << torque[0][i].item<double>() << ","; }
     // 写入估计的扭矩数据
     for(int i = 0; i < 12; ++i) { file << tau_est[0][i].item<double>() << ","; }
     // 写入关节位置数据
     for(int i = 0; i < 12; ++i) { file << joint_pos[0][i].item<double>() << ","; }
     // 写入目标关节位置数据
     for(int i = 0; i < 12; ++i) { file << joint_pos_target[0][i].item<double>() << ","; }
     // 写入关节速度数据
     for(int i = 0; i < 12; ++i) { file << joint_vel[0][i].item<double>() << ","; }
 
     file << std::endl; // 换行
 
     file.close(); // 关闭文件
 }
 