#ifndef VIO_HELLO_WORLD_PRINT_TOOLS_HPP
#define VIO_HELLO_WORLD_PRINT_TOOLS_HPP

#include <iomanip>
#include <iostream>
namespace com {
#define RED "\033[0;1;31m"
#define GREEN "\033[0;1;32m"
#define YELLOW "\033[0;1;33m"
#define BLUE "\033[0;1;34m"
#define PURPLE "\033[0;1;35m"
#define DEEPGREEN "\033[0;1;36m"
#define WHITE "\033[0;1;37m"
#define RED_IN_WHITE "\033[0;47;31m"
#define GREEN_IN_WHITE "\033[0;47;32m"
#define YELLOW_IN_WHITE "\033[0;47;33m"

#define TAIL "\033[0m"

enum class Color
{
  Red,
  Green,
  Yellow,
  Blue,
  Purple,
  DeepGreen,
  White,
  RedInWhite,
  GreenInWhite,
  YellowInWhite
};

// Template function to get colored stream
template <Color c>
std::string getColoredStream(const std::string& message) {
  if constexpr (c == Color::Red)
    return std::string(RED) + message + TAIL;
  else if constexpr (c == Color::Green)
    return std::string(GREEN) + message + TAIL;
  else if constexpr (c == Color::Yellow)
    return std::string(YELLOW) + message + TAIL;
  else if constexpr (c == Color::Blue)
    return std::string(BLUE) + message + TAIL;
  else if constexpr (c == Color::Purple)
    return std::string(PURPLE) + message + TAIL;
  else if constexpr (c == Color::DeepGreen)
    return std::string(DEEPGREEN) + message + TAIL;
  else if constexpr (c == Color::White)
    return std::string(WHITE) + message + TAIL;
  else if constexpr (c == Color::RedInWhite)
    return std::string(RED_IN_WHITE) + message + TAIL;
  else if constexpr (c == Color::GreenInWhite)
    return std::string(GREEN_IN_WHITE) + message + TAIL;
  else if constexpr (c == Color::YellowInWhite)
    return std::string(YELLOW_IN_WHITE) + message + TAIL;
}

// Utility to cause a compile-time error on unsupported types
template <class T>
struct dependent_false : std::false_type
{
};
}  // namespace com
#endif  // VIO_HELLO_WORLD_PRINT_TOOLS_HPP
