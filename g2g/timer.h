#ifndef __TIMER_H__
#define __TIMER_H__

#include <iostream>

namespace G2G {

class Timer {
 public:
  Timer(void) noexcept;
  Timer(const timespec& t) noexcept;

  void start(void) noexcept;
  void stop(void) noexcept;
  void pause(void) noexcept;
  void start_and_sync(void) noexcept;
  void stop_and_sync(void) noexcept;
  void pause_and_sync(void) noexcept;

  unsigned long getMicrosec(void) const noexcept;
  unsigned long getSec(void) const noexcept;
  double getTotal(void) const noexcept;

  bool isStarted(void) const noexcept;

  friend std::ostream& operator<<(std::ostream& o, const Timer& t);
  // to compare stopped timers
  bool operator<(const Timer& other) const;
  void print(void);
  static void sync(void);

 private:
  timespec t0, t1, res;
  bool started;
};

std::ostream& operator<<(std::ostream& o, const Timer& t);
}

#endif
