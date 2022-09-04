//
// Created by 63479 on 2022/9/3.
//

#ifndef ARM64_TEST_REGISTER_H
#define ARM64_TEST_REGISTER_H

#include "unordered_map"
#include "iostream"

enum ErrorCode {
    kNoError = 0,
    kError   = 1,
};

class Test {
public:
    explicit Test(const std::string &name) : test_name_(name) {}

    void test() const {
        std::cout     << "=====" << test_name_ << " start run=======" << std::endl;
        auto ret = run();
        if (!ret) {
            std::cout << "=====" << test_name_ << " run success=====" << std::endl;
        } else {
            std::cout << "=====" << test_name_ << " run fail========" << std::endl;
        }
    }

    virtual ErrorCode run() const = 0;

    const std::string test_name_;
};

class Container {
    friend int main(int argv, char **argc);
public:
    Container() = default;

    static Container* GetContainer() {
        static Container container;
        return &container;
    }

    void addTest(const std::string &test_name, Test *test) {
        if (!tests_.count(test_name)) {
            tests_.emplace(test_name, test);
        }
    }

    ~Container() {
        for (auto &pair : tests_) {
            delete pair.second;
            pair.second = nullptr;
        }
    }

private:
    std::unordered_map<std::string, Test*> tests_;
};

#define REGISTER_TEST(name, CLASS_NAME)                                     \
    void GlobalRegister##name() {                                           \
         Container::GetContainer()->addTest(#name, new CLASS_NAME(#name));  \
    }

#endif //ARM64_TEST_REGISTER_H
