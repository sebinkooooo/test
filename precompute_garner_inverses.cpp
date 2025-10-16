#include "optimal_primes.hpp"
#include "modular_arithmetic.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

// For k=50,000, we ONsLY store the diagonal inv[i][i]
// Storage: 50,000 × 8 bytes = 400 KB (vs 20 GB for full matrix!)

void generate_inverse_diagonal(const std::vector<u32>& primes, 
                               const std::string& output_file,
                               const std::string& array_name,
                               int chunk_size = 5000) {
    const size_t k = primes.size();
    
    std::cout << "Generating Garner inverse diagonal for k=" << k << " primes...\n";
    std::cout << "This will be written in chunks to save memory\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Open file for writing
    std::ofstream out(output_file);
    out << "#pragma once\n";
    out << "#include \"types.hpp\"\n\n";
    out << "// AUTO-GENERATED: Precomputed Garner inverse diagonal\n";
    out << "// Do not edit manually - regenerate with precompute_garner_inverses\n";
    out << "// Storage: " << (k * 8) << " bytes for k=" << k << "\n\n";
    out << "namespace PrecomputedInverses {\n\n";
    out << "// Garner inverse diagonal for k=" << k << "\n";
    out << "// inv_diag[i] = (m[0]*m[1]*...*m[i-1])^{-1} mod m[i]\n";
    out << "constexpr size_t K_" << array_name << " = " << k << ";\n";
    out << "constexpr u64 INV_DIAG_" << array_name << "[" << k << "] = {\n";
    
    // Process in chunks to avoid memory issues
    for (size_t chunk_start = 0; chunk_start < k; chunk_start += chunk_size) {
        size_t chunk_end = std::min(chunk_start + chunk_size, k);
        int chunk_num = (chunk_start / chunk_size) + 1;
        int total_chunks = (k + chunk_size - 1) / chunk_size;
        
        std::cout << "Processing chunk " << chunk_num << "/" << total_chunks 
                  << " (indices " << chunk_start << "-" << (chunk_end-1) << ")...\n";
        
        for (size_t i = chunk_start; i < chunk_end; ++i) {
            u64 inv_val;
            
            if (i == 0) {
                inv_val = 1; // Not used, but set for completeness
            } else {
                // Compute product m[0] * m[1] * ... * m[i-1] mod m[i]
                u64 prod = 1;
                for (size_t j = 0; j < i; ++j) {
                    prod = (u64)(((__uint128_t)prod * 
                                 (__uint128_t)(primes[j] % primes[i])) % 
                                 (__uint128_t)primes[i]);
                }
                
                // Compute inverse
                inv_val = modinv_u64(prod, primes[i]);
            }
            
            // Write to file
            if (i % 8 == 0) out << "    ";
            out << inv_val << "ull";
            if (i < k - 1) out << ",";
            if (i % 8 == 7 || i == k - 1) out << "\n";
            else out << " ";
            
            // Progress indicator for large computations
            if (i > 0 && (i % 1000) == 0) {
                int percent = (100 * i) / k;
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
                int eta_seconds = (elapsed * (k - i)) / i;
                std::cout << "  Progress: " << percent << "% (" << i << "/" << k 
                         << "), ETA: " << (eta_seconds / 60) << "m " << (eta_seconds % 60) << "s\n";
            }
        }
    }
    
    out << "};\n\n";
    out << "} // namespace PrecomputedInverses\n";
    out.close();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << "\n✓ Generated " << output_file << "\n";
    std::cout << "  Time: " << duration << " seconds (" << (duration / 60) << "m " << (duration % 60) << "s)\n";
    std::cout << "  Size: " << (k * 8) << " bytes (~" << ((k * 8) / 1024) << " KB)\n";
}

int main(int argc, char** argv) {
    std::cout << "=== Precomputing Garner Inverse Diagonals ===\n\n";
    
    // Generate tables for different sizes up to 50k
    std::vector<int> sizes = {
        16, 32, 64, 128, 256, 512, 1024, 2048, 5000, 10000, 25000, 50000
    };
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int k : sizes) {
        if (k > OptimalPrimes::MAX_CACHED_PRIMES) {
            std::cout << "Skipping k=" << k << " (exceeds cache of " 
                      << OptimalPrimes::MAX_CACHED_PRIMES << ")\n\n";
            continue;
        }
        
        std::cout << "\n=== Processing k=" << k << " ===\n";
        
        auto primes = OptimalPrimes::get_optimal_primes(k);
        
        double bits = 0.0;
        for (u32 p : primes) {
            bits += log2((double)p);
        }
        
        std::cout << "k=" << k << " primes => ~" << (int)bits << " bits of representation\n\n";
        
        std::string filename = "include/garner_inv_k" + std::to_string(k) + ".hpp";
        std::string array_name = std::to_string(k);
        
        generate_inverse_diagonal(primes, filename, array_name);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    
    std::cout << "\n=== All Inverse Tables Generated Successfully ===\n";
    std::cout << "Total time: " << total_duration << " seconds (" 
              << (total_duration / 60) << "m " << (total_duration % 60) << "s)\n\n";
    
    std::cout << "Storage efficiency (diagonal-only vs full matrix):\n";
    for (int k : sizes) {
        if (k > OptimalPrimes::MAX_CACHED_PRIMES) continue;
        size_t full_size = (size_t)k * k * 8;  // Full matrix
        size_t diag_size = (size_t)k * 8;       // Diagonal only
        
        if (full_size > 1024*1024) {
            std::cout << "  k=" << k << ": " << (diag_size/1024) << " KB vs " 
                      << (full_size/(1024*1024)) << " MB (saved " 
                      << (100.0 * (full_size - diag_size) / full_size) << "%)\n";
        } else {
            std::cout << "  k=" << k << ": " << diag_size << " bytes vs " 
                      << full_size << " bytes (saved " 
                      << (100.0 * (full_size - diag_size) / full_size) << "%)\n";
        }
    }
    
    std::cout << "\n=== Integration Instructions ===\n";
    std::cout << "1. Add these generated headers to your include path\n";
    std::cout << "2. Include the appropriate header based on your k value\n";
    std::cout << "3. Use garner_from_residues_fast() for zero-overhead Garner\n";
    std::cout << "4. Your CRT setup time is now ZERO - all inverses precomputed!\n\n";
    
    std::cout << "Example usage:\n";
    std::cout << "  #include \"garner_inv_k50000.hpp\"\n";
    std::cout << "  auto c = garner_from_residues_fast(r, m); // Instant!\n";
    
    return 0;
}