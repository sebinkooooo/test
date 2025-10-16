std::vector<cpp_int> generate_divisors(int M, int divisor_bits, 
    const std::unordered_set<u32>& exclude_set) {
std::vector<cpp_int> divisors;
divisors.reserve(M);

const u32 BASE_SEED_32 = 1234567u;  // Must match main.cu
const u64 BASE_SEED_64 = 1234567ull;

if (divisor_bits == 32) {
for (int i = 0; i < M; ++i) {
u32 val = divisor_at_32(BASE_SEED_32, i);
divisors.push_back(cpp_int(val));
}
} else if (divisor_bits == 64) {
for (int i = 0; i < M; ++i) {
u64 val = divisor_at_64(BASE_SEED_64, i);
divisors.push_back(cpp_int(val));
}
} else {
// For 128+ bits
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