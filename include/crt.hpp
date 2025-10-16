#pragma once
#include "types.hpp"
#include "bigint_utils.hpp"
#include <vector>

struct CRTProductTree {
    std::vector<std::vector<cpp_int>> levels;
};

CRTProductTree build_product_tree(const std::vector<u32>& m);
void remainder_tree_down(const CRTProductTree &T, const cpp_int &N, 
                        std::vector<cpp_int> &leaf_rems);
std::vector<u32> choose_moduli_dynamic(const std::vector<u32>& primes, 
                                       const cpp_int& N, int safety_bits, 
                                       int* out_k = nullptr);

// ADD THESE TWO LINES:
std::vector<u64> garner_from_residues_fast(const std::vector<u64>& r, 
                                           const std::vector<u64>& m);
std::vector<u64> garner_from_residues(const std::vector<u64>& r, 
                                      const std::vector<u64>& m);