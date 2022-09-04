//
// Created by 63479 on 2022/8/26.
//

#ifndef ARM64_TEST_DIV_MOD_CUH
#define ARM64_TEST_DIV_MOD_CUH

struct DivMod {
    DivMod(int d) {
        d_ = (d == 0) ? 1 : d;
        for (l_ = 0;; ++l_) {
            if ((1U << l_) >= d_)
                break;
        }
        uint64_t one = 1;
        uint64_t m   = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
        m_           = static_cast<uint32_t>(m);
    }

    __device__ __inline__ int Div(int idx) {
        uint32_t tm = __umulhi(m_, idx); // get high 32-bit of the product
        return (tm + idx) >> l_;
    }

    __device__ __inline__ int Mod(int idx) {
        return idx - div(idx) * d_;
    }

    __device__ __inline__ void DivMod(int idx, int &d, int &m) {
        d = Div(idx);
        m = idx - d * d_;
    }

    uint32_t d_; // divisor
    uint32_t l_; // ceil(log2(d_))
    uint32_t m_; // m' in the papaer
};

#endif //ARM64_TEST_DIV_MOD_CUH
