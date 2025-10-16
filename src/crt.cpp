// Updated crt.cpp - Fast Garner with precomputed inverses for up to 50k primes

#include "crt.hpp"
#include "modular_arithmetic.hpp"
#include "optimal_primes.hpp"

// Include precomputed inverse diagonals
#include "garner_inv_k16.hpp"
#include "garner_inv_k32.hpp"
#include "garner_inv_k64.hpp"
#include "garner_inv_k128.hpp"
#include "garner_inv_k256.hpp"
#include "garner_inv_k512.hpp"
#include "garner_inv_k1024.hpp"
#include "garner_inv_k2048.hpp"
#include "garner_inv_k5000.hpp"
#include "garner_inv_k10000.hpp"
#include "garner_inv_k25000.hpp"
#include "garner_inv_k50000.hpp"

// Fast Garner using precomputed inverse diagonal
std::vector<u64> garner_from_residues_fast(
    const std::vector<u64>& r, 
    const std::vector<u64>& m) 
{
    const size_t k = m.size();
    if (r.size() != k) {
        throw std::runtime_error("garner_fast: size mismatch");
    }

    // ADD THIS DEBUG OUTPUT:
    static bool first_call = true;
    if (first_call) {
        printf("[DEBUG] Using FAST Garner with precomputed inverses for k=%zu\n", k);
        first_call = false;
    }

    // Select the appropriate precomputed inverse diagonal
    const u64* inv_diag = nullptr;
    
    if (k <= 16) {
        inv_diag = PrecomputedInverses::INV_DIAG_16;
    } else if (k <= 32) {
        inv_diag = PrecomputedInverses::INV_DIAG_32;
    } else if (k <= 64) {
        inv_diag = PrecomputedInverses::INV_DIAG_64;
    } else if (k <= 128) {
        inv_diag = PrecomputedInverses::INV_DIAG_128;
    } else if (k <= 256) {
        inv_diag = PrecomputedInverses::INV_DIAG_256;
    } else if (k <= 512) {
        inv_diag = PrecomputedInverses::INV_DIAG_512;
    } else if (k <= 1024) {
        inv_diag = PrecomputedInverses::INV_DIAG_1024;
    } else if (k <= 2048) {
        inv_diag = PrecomputedInverses::INV_DIAG_2048;
    } else if (k <= 5000) {
        inv_diag = PrecomputedInverses::INV_DIAG_5000;
    } else if (k <= 10000) {
        inv_diag = PrecomputedInverses::INV_DIAG_10000;
    } else if (k <= 25000) {
        inv_diag = PrecomputedInverses::INV_DIAG_25000;
    } else if (k <= 50000) {
        inv_diag = PrecomputedInverses::INV_DIAG_50000;
    } else {
        throw std::runtime_error("garner_fast: k exceeds maximum precomputed size (50000)");
    }

    std::vector<u64> c(k);
    c[0] = r[0] % m[0];

    // Main Garner loop with precomputed inverses
    for (size_t i = 1; i < k; ++i) {
        u64 sum = c[0] % m[i];
        u64 prod = 1;

        // Accumulate: sum = c[0] + c[1]*m[0] + c[2]*m[0]*m[1] + ... (mod m[i])
        for (size_t j = 1; j < i; ++j) {
            prod = (u64)(((__uint128_t)prod * 
                         (__uint128_t)(m[j - 1] % m[i])) % (__uint128_t)m[i]);
            u64 term = (u64)(((__uint128_t)(c[j] % m[i]) * 
                            (__uint128_t)prod) % (__uint128_t)m[i]);
            sum = (sum + term) % m[i];
        }

        // c[i] = (r[i] - sum) * inv[i] mod m[i]
        u64 diff = (r[i] % m[i] >= sum) ? 
                   (r[i] % m[i] - sum) : 
                   (r[i] % m[i] + m[i] - sum);
        
        // Use precomputed inverse - this is the speedup!
        c[i] = (u64)(((__uint128_t)diff * 
                     (__uint128_t)inv_diag[i]) % (__uint128_t)m[i]);
    }

    return c;
}

// Backward compatible wrapper - tries fast path first, falls back to slow
std::vector<u64> garner_from_residues(
    const std::vector<u64>& r, 
    const std::vector<u64>& m) 
{
    try {
        return garner_from_residues_fast(r, m);
    } catch (const std::runtime_error& e) {
        // Fall back to computing inverses on the fly if k > 50000
        fprintf(stderr, "Warning: k=%zu exceeds precomputed tables (max 50000), "
                "computing inverses on the fly (this will be slower)\n", m.size());
        
        const size_t k = m.size();
        std::vector<u64> c(k);
        c[0] = r[0] % m[0];

        for (size_t i = 1; i < k; ++i) {
            u64 sum = c[0] % m[i];
            u64 prod = 1;

            for (size_t j = 1; j < i; ++j) {
                prod = (u64)(((__uint128_t)prod * 
                             (__uint128_t)(m[j - 1] % m[i])) % (__uint128_t)m[i]);
                u64 term = (u64)(((__uint128_t)(c[j] % m[i]) * 
                                (__uint128_t)prod) % (__uint128_t)m[i]);
                sum = (sum + term) % m[i];
            }

            u64 diff = (r[i] % m[i] >= sum) ? 
                       (r[i] % m[i] - sum) : 
                       (r[i] % m[i] + m[i] - sum);
            
            // Compute inverse on the fly (slow path)
            u64 inv_prod = 1;
            for (size_t j = 0; j < i; ++j) {
                inv_prod = (u64)(((__uint128_t)inv_prod * 
                                 (__uint128_t)(m[j] % m[i])) % (__uint128_t)m[i]);
            }
            u64 inv = modinv_u64(inv_prod, m[i]);
            
            c[i] = (u64)(((__uint128_t)diff * 
                         (__uint128_t)inv) % (__uint128_t)m[i]);
        }
        return c;
    }
}