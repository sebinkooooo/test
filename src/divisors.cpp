#include "divisors.hpp"
#include "primes.hpp"
#include <vector>

u32 divisor_at_32(u32 base_seed, int idx) {
    return (u32)(((uint64_t)idx * 104729u + base_seed) | 1u);
}

u64 divisor_at_64(u64 base_seed, int idx) {
    return (((u64)idx * 104729ull + base_seed) | 1ull);
}

std::vector<cpp_int> generate_divisors(int M, int divisor_bits, 
                                       const std::unordered_set<u32>& exclude_set) {
    std::vector<cpp_int> divisors;
    divisors.reserve(M);
    
    const u32 BASE_SEED_32 = 1234567u;
    const u64 BASE_SEED_64 = 1234567ull;
    
    // CRITICAL: Start at high offset to avoid collision with small CRT primes
    // CRT uses primes up to ~50M, so start divisors at 100M offset
    const int START_OFFSET = 100000000;
    
    if (divisor_bits == 32) {
        for (int i = 0; i < M; ++i) {
            u32 val = divisor_at_32(BASE_SEED_32, START_OFFSET + i);
            divisors.push_back(cpp_int(val));
        }
    } else if (divisor_bits == 64) {
        for (int i = 0; i < M; ++i) {
            u64 val = divisor_at_64(BASE_SEED_64, START_OFFSET + i);
            divisors.push_back(cpp_int(val));
        }
    } else {
        cpp_int base = cpp_int(1) << (divisor_bits - 1);
        for (int i = 0; i < M; ++i) {
            cpp_int offset = cpp_int(i * 104729 + 1234567);
            cpp_int divisor = base + offset;
            if (divisor % 2 == 0) divisor += 1;
            divisors.push_back(divisor);
        }
    }
    return divisors;
}