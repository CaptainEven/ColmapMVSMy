#include "threading.h"


namespace colmap {

ThreadPool::ThreadPool(const int num_threads)
    : stopped_(false), num_active_workers_(0) {
  const int num_effective_threads = GetEffectiveNumThreads(num_threads);
  for (int index = 0; index < num_effective_threads; ++index) {
    std::function<void(void)> worker =
        std::bind(&ThreadPool::WorkerFunc, this, index);
    workers_.emplace_back(worker);
  }
}

ThreadPool::~ThreadPool() { Stop(); }

void ThreadPool::Stop() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
      return;
    }
    stopped_ = true;
  }

  {
    std::queue<std::function<void()>> empty_tasks;
    std::swap(tasks_, empty_tasks);
  }

  task_condition_.notify_all();

  for (auto& worker : workers_) {
    worker.join();
  }

  finished_condition_.notify_all();
}

void ThreadPool::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!tasks_.empty() || num_active_workers_ > 0) {
    finished_condition_.wait(
        lock, [this]() { return tasks_.empty() && num_active_workers_ == 0; });
  }
}

void ThreadPool::WorkerFunc(const int index) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    thread_id_to_index_.emplace(GetThreadId(), index);
  }

  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      task_condition_.wait(lock,
                           [this] { return stopped_ || !tasks_.empty(); });
      if (stopped_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
      num_active_workers_ += 1;
    }

    task();

    {
      std::unique_lock<std::mutex> lock(mutex_);
      num_active_workers_ -= 1;
    }

    finished_condition_.notify_all();
  }
}

std::thread::id ThreadPool::GetThreadId() const {
  return std::this_thread::get_id();
}

int ThreadPool::GetThreadIndex() {
  std::unique_lock<std::mutex> lock(mutex_);
  return thread_id_to_index_.at(GetThreadId());
}

int GetEffectiveNumThreads(const int num_threads) {
  int num_effective_threads = num_threads;
  if (num_threads <= 0) {
    num_effective_threads = std::thread::hardware_concurrency();
  }

  if (num_effective_threads <= 0) {
    num_effective_threads = 1;
  }

  return num_effective_threads;
}

}  // namespace colmap
