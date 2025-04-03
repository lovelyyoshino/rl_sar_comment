/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

 #include "observation_buffer.hpp"

 // ObservationBuffer类用于存储和管理观察数据的缓冲区
 ObservationBuffer::ObservationBuffer() {}
 
 ObservationBuffer::ObservationBuffer(int num_envs,
                                      int num_obs,
                                      int include_history_steps)
     : num_envs(num_envs),
       num_obs(num_obs),
       include_history_steps(include_history_steps)
 {
     // 计算总观察数
     num_obs_total = num_obs * include_history_steps;
     // 初始化观察缓冲区，大小为(num_envs, num_obs_total)，数据类型为float32
     obs_buf = torch::zeros({num_envs, num_obs_total}, torch::dtype(torch::kFloat32));
 }
 
 // 重置指定索引的观察数据
 void ObservationBuffer::reset(std::vector<int> reset_idxs, torch::Tensor new_obs)
 {
     std::vector<torch::indexing::TensorIndex> indices; // 用于存储索引
     for (int idx : reset_idxs)
     {
         indices.push_back(torch::indexing::Slice(idx)); // 将重置索引添加到索引列表中
     }
     // 更新观察缓冲区，重复新观察数据以填充历史步骤,num_obs对应reset_idxs数据，然后每个indices是重复传入数据，
     // 这里index_put_()函数是根据indices索引，将new_obs的数据赋值给obs_buf，传入的是vector的原因是reset_idxs可能包含多个索引
     obs_buf.index_put_(indices, new_obs.repeat({1, include_history_steps}));
 }
 
 // 插入新的观察数据，删除前面的，保留最后面的
 void ObservationBuffer::insert(torch::Tensor new_obs)
 {
     // 将观察数据向后移动，这里的index里面，torch::indexing::Slice(torch::indexing::None)表示所有行，torch::indexing::Slice(num_obs, num_obs * include_history_steps)表示从num_obs到num_obs * include_history_steps的列
     torch::Tensor shifted_obs = obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(num_obs, num_obs * include_history_steps)}).clone();
     // 更新观察缓冲区，保留最新的历史数据，这个意思就是保留最新的num_obs * (include_history_steps - 1)个数据，然后从shifted_obs中取num_obs个数据
     obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, num_obs * (include_history_steps - 1))}) = shifted_obs;
 
     // 添加新的观察数据到缓冲区，这个意思就是从new_obs中取num_obs个数据，然后添加到缓冲区中，这里-num_obs表示从最后一列开始，torch::indexing::None表示最后一列
     obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(-num_obs, torch::indexing::None)}) = new_obs;
 }
 
 /**
  * @brief 获取由obs_ids索引的观察历史。
  *
  * @param obs_ids 一个整数数组，用于索引所需的观察数据，其中0是最新观察，include_history_steps - 1是最旧观察。
  * @return 返回一个包含连接观察数据的torch::Tensor。
  */
 torch::Tensor ObservationBuffer::get_obs_vec(std::vector<int> obs_ids)
 {
     std::vector<torch::Tensor> obs; // 用于存储观察数据的向量
     for (int i = 0; i < obs_ids.size(); ++i)
     {
         int obs_id = obs_ids[i]; // 当前观察ID
         int slice_idx = include_history_steps - obs_id - 1; // 计算切片索引
         // 从观察缓冲区中提取对应的观察数据，一行都需要拿到，所以是torch::indexing::Slice(torch::indexing::None)，然后从slice_idx * num_obs到(slice_idx + 1) * num_obs的列
         obs.push_back(obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(slice_idx * num_obs, (slice_idx + 1) * num_obs)}));
     }
     // 将提取的观察数据连接成一个Tensor并返回
     return torch::cat(obs, -1);
 }
 