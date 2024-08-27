#include <filesystem>
#include <iostream>
#include <vector>
namespace fs = std::__fs::filesystem;

#define PRINT1(a) std::cout <<  a << '\n'
#define PRINT2(a, b) std::cout <<  a << ", " << b <<'\n'
#define PRINT3(a, b, c) std::cout <<  a << ", " << b << ", " << c <<'\n'
#define PRINT4(a, b, c, d) std::cout <<  a << ", " << b << ", " << c << ", " << d <<'\n'

#define VA_GENERIC(_1, _2, _3, _4, x, ...) x
#define PRINT(...) VA_GENERIC(__VA_ARGS__, PRINT4, PRINT3, PRINT2, PRINT1)(__VA_ARGS__)

#define SHAPE(x) std::cout << x.rows() << " X " << x.cols() << '\n'
#define CVSH(x) std::cout << x.rows << " X " << x.cols << " X " << x.channels() << '\n'
#define DTYPE double_t
#define CVTYPE(x) PRINT(cv::typeToString(x.type()))


int get_num(const std::string& s);
bool compare(const fs::path a, const fs::path b);
bool check_file_extensions(const fs::path& file);

