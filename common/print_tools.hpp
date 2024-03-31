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

inline void printHelloWorldVIO() {
  std::cout << BLUE  << std::endl
            << "  _   _      _ _        __        __         _     _  __     _____ ___  \n"
               " | | | | ___| | | ___   \\ \\      / ___  _ __| | __| | \\ \\   / |_ _/ _ \\ \n"
               " | |_| |/ _ | | |/ _ \\   \\ \\ /\\ / / _ \\| '__| |/ _` |  \\ \\ / / | | | | |\n"
               " |  _  |  __| | | (_) |   \\ V  V | (_) | |  | | (_| |   \\ V /  | | |_| |\n"
               " |_| |_|\\___|_|_|\\___/     \\_/\\_/ \\___/|_|  |_|\\__,_|    \\_/  |___\\___/ \n"
            << TAIL << std::endl;
}

inline void printZJU() {
  std::cout
      << PURPLE
      << "       _____                   _____                   _____          \n"
         "         /\\    \\                 /\\    \\                 /\\    \\         \n"
         "        /::\\    \\               /::\\    \\               /::\\____\\        \n"
         "        \\:::\\    \\              \\:::\\    \\             /:::/    /        \n"
         "         \\:::\\    \\              \\:::\\    \\           /:::/    /         \n"
         "          \\:::\\    \\              \\:::\\    \\         /:::/    /          \n"
         "           \\:::\\    \\              \\:::\\    \\       /:::/    /           \n"
         "            \\:::\\    \\             /::::\\    \\     /:::/    /            \n"
         "             \\:::\\    \\   _____   /::::::\\    \\   /:::/    /      _____  \n"
         "              \\:::\\    \\ /\\    \\ /:::/\\:::\\    \\ /:::/____/      /\\    \\ \n"
         "_______________\\:::\\____/::\\    /:::/  \\:::\\____|:::|    /      /::\\____\\\n"
         "\\::::::::::::::::::/    \\:::\\  /:::/    \\::/    |:::|____\\     /:::/    /\n"
         " \\::::::::::::::::/____/ \\:::\\/:::/    / \\/____/ \\:::\\    \\   /:::/    / \n"
         "  \\:::\\~~~~\\~~~~~~        \\::::::/    /           \\:::\\    \\ /:::/    /  \n"
         "   \\:::\\    \\              \\::::/    /             \\:::\\    /:::/    /   \n"
         "    \\:::\\    \\              \\::/    /               \\:::\\__/:::/    /    \n"
         "     \\:::\\    \\              \\/____/                 \\::::::::/    /     \n"
         "      \\:::\\    \\                                      \\::::::/    /      \n"
         "       \\:::\\____\\                                      \\::::/    /       \n"
         "        \\::/    /                                       \\::/____/        \n"
         "         \\/____/                                         ~~              \n"
         "                                                                         "
      << TAIL << std::endl;
}

inline void printZJUV1() {
  std::cout << PURPLE
            << "                                                                     \n"
               "                                                           \n"
               "ZZZZZZZZZZZZZZZZZZZ         JJJJJJJJJJUUUUUUUU     UUUUUUUU\n"
               "Z:::::::::::::::::Z         J:::::::::U::::::U     U::::::U\n"
               "Z:::::::::::::::::Z         J:::::::::U::::::U     U::::::U\n"
               "Z:::ZZZZZZZZ:::::Z          JJ:::::::JUU:::::U     U:::::UU\n"
               "ZZZZZ     Z:::::Z             J:::::J  U:::::U     U:::::U \n"
               "        Z:::::Z               J:::::J  U:::::D     D:::::U \n"
               "       Z:::::Z                J:::::J  U:::::D     D:::::U \n"
               "      Z:::::Z                 J:::::j  U:::::D     D:::::U \n"
               "     Z:::::Z                  J:::::J  U:::::D     D:::::U \n"
               "    Z:::::Z       JJJJJJJ     J:::::J  U:::::D     D:::::U \n"
               "   Z:::::Z        J:::::J     J:::::J  U:::::D     D:::::U \n"
               "ZZZ:::::Z     ZZZZJ::::::J   J::::::J  U::::::U   U::::::U \n"
               "Z::::::ZZZZZZZZ:::J:::::::JJJ:::::::J  U:::::::UUU:::::::U \n"
               "Z:::::::::::::::::ZJJ:::::::::::::JJ    UU:::::::::::::UU  \n"
               "Z:::::::::::::::::Z  JJ:::::::::JJ        UU:::::::::UU    \n"
               "ZZZZZZZZZZZZZZZZZZZ    JJJJJJJJJ            UUUUUUUUU      \n"
               "                                                           \n"
               "                                                           \n"
            << TAIL << std::endl;
}

inline void printZJUV2() {
  std::cout << PURPLE
            << " _______    _ _    _ \n"
               " |___  /   | | |  | |\n"
               "    / /    | | |  | |\n"
               "   / / _   | | |  | |\n"
               "  / /_| |__| | |__| |\n"
               " /_____\\____/ \\____/ \n"
               "                     \n"
               "                     "
            << TAIL << std::endl;
}

inline void printFZ() {
  std::cout << GREEN
               "                       _oo0oo_                      \n"
               "                      o8888888o                     \n"
               "                      88\" . \"88                     \n"
               "                      (| -_- |)                     \n"
               "                      0\\  =  /0                     \n"
               "                   ___/‘---’\\___                   \n"
               "                  .' \\|       |/ '.                 \n"
               "                 / \\\\|||  :  |||// \\                \n"
               "                / _||||| -卍-|||||_ \\               \n"
               "               |   | \\\\\\  -  /// |   |              \n"
               "               | \\_|  ''\\---/''  |_/ |              \n"
               "               \\  .-\\__  '-'  ___/-. /              \n"
               "             ___'. .'  /--.--\\  '. .'___            \n"
               "         .\"\" ‘<  ‘.___\\_<|>_/___.’>’ \"\".          \n"
               "       | | :  ‘- \\‘.;‘\\ _ /’;.’/ - ’ : | |        \n"
               "         \\  \\ ‘_.   \\_ __\\ /__ _/   .-’ /  /        \n"
               "    =====‘-.____‘.___ \\_____/___.-’___.-’=====     \n"
               "                       ‘=---=’                      \n"
               "                                                    \n"
               "...Best wishes for you, never fried chicken, never bugs...\n"
               ".......................Amitabha......................\n"
            << TAIL << std::endl;
}

inline void printKeyboard() {
  std::cout << BLUE
            << "   ┌─────────────────────────────────────────────────────────────┐\n"
               "   │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│\n"
               "   ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\\ │`~ ││\n"
               "   │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│\n"
               "   ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││\n"
               "   │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│\n"
               "   ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│\" '│ Enter  ││\n"
               "   │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│\n"
               "   ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││\n"
               "   │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│\n"
               "   │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │\n"
               "   │      └───┴─────┴───────────────────────┴─────┴───┘          │\n"
               "   └─────────────────────────────────────────────────────────────┘\n"
            << TAIL << std::endl;
}

inline void printCNM() {
  std::cout << PURPLE
            << "  ┏┓　　　┏┓\n"
               "  ┏┛┻━━━┛┻┓\n"
               "  ┃　　　　　　  ┃\n"
               "  ┃　　　━　　　 ┃\n"
               "  ┃　＞　　　＜　┃\n"
               "  ┃　　　　　　　┃\n"
               "  ┃...　⌒　...  ┃\n"
               "  ┃　　　　　　　┃\n"
               "  ┗━┓　　　┏━┛\n"
               "      ┃　　　┃　\n"
               "      ┃　　　┃\n"
               "      ┃　　　┃\n"
               "      ┃　　　┃  Best Wishes\n"
               "      ┃　　　┃  Never BUG!\n"
               "      ┃　　　┃\n"
               "      ┃　　　┗━━━┓\n"
               "      ┃　　　　　　　┣┓\n"
               "      ┃　　　　　　　┏┛\n"
               "      ┗┓┓┏━┳┓┏┛\n"
               "        ┃┫┫　┃┫┫\n"
               "        ┗┻┛　┗┻┛"
            << TAIL << std::endl;
}

inline void printGirl() {
  std::cout << GREEN
            << "\n"
               "                         .::::.\n"
               "                       .::::::::.\n"
               "                      :::::::::::\n"
               "                   ..:::::::::::'\n"
               "                '::::::::::::'\n"
               "                  .::::::::::\n"
               "             '::::::::::::::..\n"
               "                  ..::::::::::::.\n"
               "                ``::::::::::::::::\n"
               "                 ::::``:::::::::'        .:::.\n"
               "                ::::'   ':::::'       .::::::::.\n"
               "              .::::'      ::::     .:::::::'::::.\n"
               "             .:::'       :::::  .:::::::::' ':::::.\n"
               "            .::'        :::::.:::::::::'      ':::::.\n"
               "           .::'         ::::::::::::::'         ``::::.\n"
               "       ...:::           ::::::::::::'              ``::.\n"
               "      ````':.          ':::::::::'                  ::::..\n"
               "                         '.:::::'                    ':'````..\n"
               " "
            << TAIL << std::endl;
}
}  // namespace com
#endif  // VIO_HELLO_WORLD_PRINT_TOOLS_HPP
