#include <thread>
#include <iostream>
#include <mutex>
#include <chrono>
#include <condition_variable>

class KalmanFilter {
public:
  KalmanFilter();
  void process(int v) {
    std::lock_guard<std::mutex> g(mw);
    it_work_todo = v;
  }

  int predict(int v);

  void work_process() {
    int this_worker;
    while (true) {
      {
        std::lock_guard<std::mutex> g(mw);
        this_worker = it_work_todo;
        it_work_todo++;
      }
      std::cout << "processing " << this_worker << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      std::cout << "processing " << this_worker << "ended." << std::endl;
      it_work_done = this_worker;
      {
        std::unique_lock<std::mutex> lk(m_worker);
        cv_worker.notify_all();
      }
    }
  }

  int counter = 0;

  std::thread tworker;
  int it_work_todo; // iterators are safe against map modifications    http://kera.name/articles/2011/06/iterator-invalidation-rules-c0x/
  int it_work_done;


  std::mutex mw;

  std::condition_variable cv_worker;
  std::mutex m_worker;
};

KalmanFilter::KalmanFilter() : tworker(&KalmanFilter::work_process, this) {
  it_work_done = 0;
  it_work_todo = 1;
}

void defcall(int ms, KalmanFilter& kf, int v) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  kf.process(v);
}

void defpredict(int ms, KalmanFilter& kf, int v) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  kf.predict(v);
}

int KalmanFilter::predict(int v) {
  std::cout << "predict waiting" << v << std::endl;
  std::unique_lock<std::mutex> lk(m_worker);
  if (it_work_done<v) {
    cv_worker.wait(lk, [&]{return it_work_done >= v;});
  }
  std::cout << "predict waiting" << v << " done." << std::endl;
  return 1;
}

int main ( int argc , char *argv[] ) {

  KalmanFilter kf;

  kf.process(3);

  std::thread ta(defpredict, 0, std::ref(kf), 20);
  std::thread tb(defpredict, 100, std::ref(kf), 2);
  std::thread tc(defpredict, 200, std::ref(kf), 1);

  std::cout << kf.predict(-5) << std::endl;

  std::thread t1(defcall, 2000, std::ref(kf), 1);
  std::thread t2(defcall, 6000, std::ref(kf), 10);
  std::thread t3(defcall, 10000, std::ref(kf), 1);


  std::cout << kf.predict(12) << std::endl;

  std::this_thread::sleep_for(std::chrono::milliseconds(1000000));

  return 1;
}
