#ifndef VIO_HELLO_WORLD_WALL_TIME_HPP
#define VIO_HELLO_WORLD_WALL_TIME_HPP
#include <stdio.h>
#include <sys/time.h>

namespace com {

double WallTimeInSeconds()
{
  timeval time_val;
  gettimeofday(&time_val, NULL);
  return (time_val.tv_sec + time_val.tv_usec * 1e-6);
}

}  // namespace com

#endif  // VIO_HELLO_WORLD_WALL_TIME_HPP
