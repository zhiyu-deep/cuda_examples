//
// Created by 63479 on 2022/9/3.
//

#include "register.h"


extern void GlobalRegisterGemm();
extern void GlobalRegisterTranspose();
extern void GlobalRegisterScan();
extern void GlobalRegisterConv();

bool GlobalRegister() {
    GlobalRegisterGemm();
    GlobalRegisterTranspose();
    GlobalRegisterScan();
    GlobalRegisterConv();

    return true;
}

auto register_status = GlobalRegister();
