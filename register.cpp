//
// Created by 63479 on 2022/9/3.
//

#include "register.h"


extern void GlobalRegisterGemm();

bool GlobalRegister() {
    GlobalRegisterGemm();
    return true;
}

auto register_status = GlobalRegister();
