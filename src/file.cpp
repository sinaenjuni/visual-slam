#include "file.h"

int get_num(const std::string& s)
{
    int num=0;
    for(char c: s)
        if(c>47 && c<58)
            num=num*10+c-48;
    return num;
}

bool compare(const fs::path a, const fs::path b)
{
    return get_num(a.stem().string()) < get_num(b.stem().string());
}

bool check_file_extensions(const fs::path& file)
{
    std::vector<std::string> extensions = { ".png", ".jpeg", ".jpg" };
    return *std::find(extensions.begin(), extensions.end(), file.extension()) == file.extension();
}