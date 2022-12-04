//
// Created by 63479 on 2022/9/3.
//

#include "register.h"


extern void GlobalRegisterGemm();
extern void GlobalRegisterTranspose();
extern void GlobalRegisterScan();
extern void GlobalRegisterConv();
extern void GlobalRegisterWmma();
extern void GlobalRegisterPtxsGemm();

bool GlobalRegister() {
    GlobalRegisterGemm();
    GlobalRegisterTranspose();
    GlobalRegisterScan();
    GlobalRegisterConv();
    GlobalRegisterWmma();
    GlobalRegisterPtxsGemm();

    return true;
}

auto register_status = GlobalRegister();
