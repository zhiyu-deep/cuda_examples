#include "register.h"

int main(int argv, char **argc) {
    const Container *container = Container::GetContainer();

    if (argv > 1) {
        std::string op_info(argc[1]);
        if (op_info != "all") {
            if (container->tests_.count(op_info)) {
                container->tests_.at(op_info)->test();
            }
            return 0;
        }
    }
    for (const auto &pair : container->tests_) {
        pair.second->test();
    }
    return 0;
}
