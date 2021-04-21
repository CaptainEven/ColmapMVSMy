#ifndef COLMAP_SRC_UTIL_THREADING_
#define COLMAP_SRC_UTIL_THREADING_

#include <atomic>
#include <climits>
#include <functional>
#include <future>
#include <list>
#include <queue>
#include <unordered_map>


namespace colmap {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif

// Define `thread_local` cross-platform.
#ifndef thread_local
#if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
#define thread_local _Thread_local
#elif defined _WIN32 && (defined _MSC_VER || defined __ICL || \
                         defined __DMC__ || defined __BORLANDC__)
#define thread_local __declspec(thread)
#elif defined __GNUC__ || defined __SUNPRO_C || defined __xlC__
#define thread_local __thread
#else
#error "Cannot define thread_local"
#endif
#endif

#ifdef __clang__
#pragma clang diagnostic pop  // -Wkeyword-macro
#endif


// A thread pool class to submit generic tasks (functors) to a pool of workers:
//
//    ThreadPool thread_pool;
//    thread_pool.AddTask([]() { /* Do some work */ });
//    auto future = thread_pool.AddTask([]() { /* Do some work */ return 1; });
//    const auto result = future.get();
//    for (int i = 0; i < 10; ++i) {
//      thread_pool.AddTask([](const int i) { /* Do some work */ });
//    }
//    thread_pool.Wait();
//
class ThreadPool {
 public:
  static const int kMaxNumThreads = -1;

  explicit ThreadPool(const int num_threads = kMaxNumThreads);
  ~ThreadPool();

  inline size_t NumThreads() const;

  // Add new task to the thread pool.
  template <class func_t, class... args_t>
  auto AddTask(func_t&& f, args_t&&... args)
      -> std::future<typename std::result_of<func_t(args_t...)>::type>;

  // Stop the execution of all workers.
  void Stop();

  // Wait until tasks are finished.
  void Wait();

  // Get the unique identifier of the current thread.
  std::thread::id GetThreadId() const;

  // Get the index of the current thread. In a thread pool of size N,
  // the thread index defines the 0-based index of the thread in the pool.
  // In other words, there are the thread indices 0, ..., N-1.
  int GetThreadIndex();

 private:
  void WorkerFunc(const int index);

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex mutex_;
  std::condition_variable task_condition_;
  std::condition_variable finished_condition_;

  bool stopped_;
  int num_active_workers_;

  std::unordered_map<std::thread::id, int> thread_id_to_index_;
};


// Return the number of logical CPU cores if num_threads <= 0,
// otherwise return the input value of num_threads.
int GetEffectiveNumThreads(const int num_threads);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t ThreadPool::NumThreads() const { return workers_.size(); }

template <class func_t, class... args_t>
auto ThreadPool::AddTask(func_t&& f, args_t&&... args)
    -> std::future<typename std::result_of<func_t(args_t...)>::type> {
  typedef typename std::result_of<func_t(args_t...)>::type return_t;

  auto task = std::make_shared<std::packaged_task<return_t()>>(
      std::bind(std::forward<func_t>(f), std::forward<args_t>(args)...));

  std::future<return_t> result = task->get_future();

  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
      throw std::runtime_error("Cannot add task to stopped thread pool.");
    }
    tasks_.emplace([task]() { (*task)(); });
  }

  task_condition_.notify_one();

  return result;
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_THREADING_
