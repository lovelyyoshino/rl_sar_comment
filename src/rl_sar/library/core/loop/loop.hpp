/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

 #ifndef LOOP_H
 #define LOOP_H
 
 #include <iostream>
 #include <thread>
 #include <chrono>
 #include <functional>
 #include <mutex>
 #include <condition_variable>
 #include <atomic>
 #include <vector>
 #include <sstream>
 #include <iomanip>
 
 // LoopFunc类用于创建一个周期性执行的函数循环
 class LoopFunc
 {
 public:
     // 构造函数，初始化循环函数的名称、周期、执行函数和绑定的CPU
     LoopFunc(const std::string &name, double period, std::function<void()> func, int bindCPU = -1)
         : _name(name), _period(period), _func(func), _bindCPU(bindCPU), _running(false) {}
 
     // 启动循环函数
     void start()
     {
         _running = true; // 设置运行状态为true
         log("[Loop Start] named: " + _name + ", period: " + formatPeriod() + "(ms)" + (_bindCPU != -1 ? ", run at cpu: " + std::to_string(_bindCPU) : ", cpu unspecified"));
         
         // 如果指定了CPU，则创建线程并设置线程亲和性
         if (_bindCPU != -1)
         {
             _thread = std::thread(&LoopFunc::loop, this);
             setThreadAffinity(_thread.native_handle(), _bindCPU);
         }
         else // 否则只创建线程
         {
             _thread = std::thread(&LoopFunc::loop, this);
         }
         _thread.detach(); // 分离线程，使其在后台运行
     }
 
     // 关闭循环函数
     void shutdown()
     {
         {
             std::unique_lock<std::mutex> lock(_mutex); // 锁定互斥量
             _running = false; // 设置运行状态为false
             _cv.notify_one(); // 通知等待的线程
         }
         if (_thread.joinable()) // 如果线程可连接
         {
             _thread.join(); // 等待线程结束
         }
         log("[Loop End] named: " + _name); // 记录结束日志
     }
 
 private:
     std::string _name; // 循环函数的名称
     double _period; // 循环周期（秒）
     std::function<void()> _func; // 循环执行的函数
     int _bindCPU; // 绑定的CPU核心
     std::atomic<bool> _running; // 运行状态
     std::mutex _mutex; // 互斥量，用于保护共享资源
     std::condition_variable _cv; // 条件变量，用于线程同步
     std::thread _thread; // 线程对象
 
     // 循环执行函数
     void loop()
     {
         while (_running) // 当运行状态为true时持续循环
         {
             auto start = std::chrono::steady_clock::now(); // 记录开始时间
 
             _func(); // 执行传入的函数
 
             auto end = std::chrono::steady_clock::now(); // 记录结束时间
             auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); // 计算执行时间
             auto sleepTime = std::chrono::milliseconds(static_cast<int>((_period * 1000) - elapsed.count())); // 计算休眠时间
             
             if (sleepTime.count() > 0) // 如果休眠时间大于0
             {
                 std::unique_lock<std::mutex> lock(_mutex); // 锁定互斥量
                 // 等待休眠时间或被通知
                 if (_cv.wait_for(lock, sleepTime, [this]
                                  { return !_running; }))
                 {
                     break; // 如果运行状态变为false，则退出循环
                 }
             }
         }
     }
 
     // 格式化周期为字符串（毫秒）
     std::string formatPeriod() const
     {
         std::ostringstream stream;
         stream << std::fixed << std::setprecision(0) << _period * 1000; // 将周期转换为毫秒并格式化
         return stream.str();
     }
 
     // 记录日志信息
     void log(const std::string &message)
     {
         static std::mutex logMutex; // 静态互斥量，保护日志输出
         std::lock_guard<std::mutex> lock(logMutex); // 自动锁定互斥量
         std::cout << message << std::endl; // 输出日志信息
     }
 
     // 设置线程的CPU亲和性
     void setThreadAffinity(std::thread::native_handle_type threadHandle, int cpuId)
     {
         cpu_set_t cpuset; // CPU集合
         CPU_ZERO(&cpuset); // 清空CPU集合
         CPU_SET(cpuId, &cpuset); // 将指定的CPU添加到集合中
         // 设置线程亲和性
         if (pthread_setaffinity_np(threadHandle, sizeof(cpu_set_t), &cpuset) != 0)
         {
             std::ostringstream oss;
             oss << "Error setting thread affinity: CPU " << cpuId << " may not be valid or accessible.";
             throw std::runtime_error(oss.str()); // 抛出异常
         }
     }
 };
 
 #endif // LOOP_H
 