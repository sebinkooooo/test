#include "types.hpp"
#include "timing.hpp"
#include "bigint_utils.hpp"
#include "primes.hpp"
#include "crt.hpp"
#include "divisors.hpp"
#include "cuda_utils.hpp"
#include "gpu_kernels.cuh"
#include "warmup.cuh"

#ifndef NO_CGBN
#include "cgbn_utils.hpp"
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <limits>

#ifndef NO_CGBN
__global__ void cgbn_divrem_kernel(cgbn_error_report_t *report,
                                   const cgbn_bn_mem_t *d_N_single,
                                   cgbn_bn_mem_t *divs,
                                   cgbn_bn_mem_t *qouts,
                                   cgbn_bn_mem_t *routs,
                                   int count);
#endif


void print_usage(const char* prog_name) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s N M1 [M2 M3 ...]\n", prog_name);
    fprintf(stderr, "  %s N -k K M1 [M2 M3 ...]   (explicitly set number of CRT moduli)\n", prog_name);
    fprintf(stderr, "\nDivisor size: DIVISOR_BITS=%d (compile with -DDIVISOR_BITS=X)\n", DIVISOR_BITS);
    fprintf(stderr, "CGBN_BITS=%d (compile with -DCGBN_BITS=X for larger numbers)\n", CGBN_BITS);
}


void check_gpu_capability() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 7) {
        std::cerr << "Warning: device architecture may not support 128-bit math\n";
    }
}

struct SetupData {
    std::vector<u32> m;
    std::vector<u32> r;
    std::vector<u32> c32;
    int k_used;
    double choose_ms;
    double residues_ms;
    double garner_ms;
};

SetupData perform_crt_setup(const cpp_int& N, int k, int safety_bits) {
    SetupData setup;
    
    auto t_choose_start = now_tp();
    if (k < 0) {
        int k_dyn = 0;
        setup.m = choose_moduli_dynamic(global_primes, N, safety_bits, &k_dyn);
        setup.k_used = k_dyn;
    } else {
        if (k > (int)global_primes.size()) {
            setup.k_used = (int)global_primes.size();
        } else {
            setup.k_used = k;
        }
        setup.m.assign(global_primes.begin(), global_primes.begin() + setup.k_used);
    }
    setup.choose_ms = ms_since(t_choose_start);

    auto t_residues_start = now_tp();
    setup.r.resize(setup.k_used);
#if FAST_REMAINDER_TREE
    CRTProductTree PT = build_product_tree(setup.m);
    std::vector<cpp_int> leaf_rems;
    remainder_tree_down(PT, N, leaf_rems);
    for (int i = 0; i < setup.k_used; ++i) {
        setup.r[i] = (u32)leaf_rems[i];
    }
#else
    for (int i = 0; i < setup.k_used; ++i) {
        setup.r[i] = (u32)(N % cpp_int(setup.m[i]));
    }
#endif
    setup.residues_ms = ms_since(t_residues_start);

    auto t_garner_start = now_tp();
    std::vector<u64> c = garner_from_residues(
        std::vector<u64>(setup.r.begin(), setup.r.end()),
        std::vector<u64>(setup.m.begin(), setup.m.end())
    );
    setup.c32.assign(c.begin(), c.end());
    setup.garner_ms = ms_since(t_garner_start);

    return setup;
}

struct DeviceCRTData {
    u32 *d_c;
    u32 *d_m;
    double h2d_ms;
    
    DeviceCRTData() : d_c(nullptr), d_m(nullptr), h2d_ms(0.0) {}
    
    void allocate_and_upload(const std::vector<u32>& c32, 
                            const std::vector<u32>& m, 
                            int k_used) {
        auto t_h2d_start = now_tp();
        
        // Allocate pinned host buffers
        u32 *h_c = nullptr, *h_m = nullptr;
        cudaHostAlloc(&h_c, k_used * sizeof(u32), cudaHostAllocDefault);
        cudaHostAlloc(&h_m, k_used * sizeof(u32), cudaHostAllocDefault);
        memcpy(h_c, c32.data(), k_used * sizeof(u32));
        memcpy(h_m, m.data(), k_used * sizeof(u32));
        
        // Allocate device buffers
        CUDA_CHECK(cudaMalloc(&d_c, k_used * sizeof(u32)));
        CUDA_CHECK(cudaMalloc(&d_m, k_used * sizeof(u32)));
        
        // Fast H2D transfers
        CUDA_CHECK(cudaMemcpy(d_c, h_c, k_used * sizeof(u32), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m, h_m, k_used * sizeof(u32), cudaMemcpyHostToDevice));
        
        h2d_ms = ms_since(t_h2d_start);
        
        // Free pinned host buffers
        cudaFreeHost(h_c);
        cudaFreeHost(h_m);
    }
    
    void free() {
        if (d_c) CUDA_CHECK(cudaFree(d_c));
        if (d_m) CUDA_CHECK(cudaFree(d_m));
        d_c = nullptr;
        d_m = nullptr;
    }
    
    ~DeviceCRTData() {
        free();
    }
};

struct BenchmarkResults {
    double pgen_ms;
    double h2d_chunks_ms;
    double kernel_ms;
    double d2h_chunks_ms;
    double total_gpu_ms;
    double total_cpu_ms;
    int total_mism;
    
    BenchmarkResults() : pgen_ms(0), h2d_chunks_ms(0), kernel_ms(0), 
                        d2h_chunks_ms(0), total_gpu_ms(0), total_cpu_ms(0), 
                        total_mism(0) {}
};

void run_crt_benchmark_32(int M, const cpp_int& N, 
                         const std::vector<cpp_int>& divisors,
                         const DeviceCRTData& crt_data, int k_used,
                         BenchmarkResults& results,
                         const u64 BASE_SEED_64, const u32 BASE_SEED_32) {
    const int CHUNK_SIZE = 10000000;
    
    // Allocate device buffers
    u32 *d_P = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_P, M * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_out, M * sizeof(u32)));
    
    if (GPU_GENERATE_DIVISORS) {
        int threads = 256;
        int blocks = (M + threads - 1) / threads;
        // CRITICAL: Use same offset as CPU (100M) to avoid CRT prime collisions
        const int START_OFFSET = 100000000;
        u32 offset_seed = BASE_SEED_32 + (u32)((uint64_t)START_OFFSET * 104729ull);
        generate_divisors_kernel_32<<<blocks, threads>>>(offset_seed, M, d_P);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        std::vector<u32> P_cpu(M);
        for (int i = 0; i < M; ++i) {
            P_cpu[i] = (u32)divisors[i];
        }
        CUDA_CHECK(cudaMemcpy(d_P, P_cpu.data(), M * sizeof(u32), cudaMemcpyHostToDevice));
    }
    
    std::vector<u32> gpu_chunk, cpu_chunk;
    
    for (int offset = 0; offset < M; offset += CHUNK_SIZE) {
        int chunk = std::min(CHUNK_SIZE, M - offset);
        u32 *d_P_chunk = d_P + offset;
        u32 *d_out_chunk = d_out + offset;
        
        // Kernel timing
        int threads = 256;
        int blocks = (chunk + threads - 1) / threads;
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        
        remainders_via_crt_32<<<blocks, threads>>>(
            d_P_chunk, d_out_chunk, chunk, crt_data.d_c, crt_data.d_m, k_used
        );
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        
        float gpu_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t0, t1));
        results.kernel_ms += gpu_ms;
        results.total_gpu_ms += gpu_ms;
        
        // D2H timing
        auto t_d2h_start = now_tp();
        gpu_chunk.resize(chunk);
        CUDA_CHECK(cudaMemcpy(gpu_chunk.data(), d_out_chunk, 
                             chunk * sizeof(u32), cudaMemcpyDeviceToHost));
        double d2h_time = ms_since(t_d2h_start);
        results.d2h_chunks_ms += d2h_time;
        results.total_gpu_ms += d2h_time; 
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
        
        // CPU compute
        auto t_cpu_start = now_tp();
        cpu_chunk.resize(chunk);
        if (GPU_GENERATE_DIVISORS) {
            for (int i = 0; i < chunk; ++i) {
                u32 p = divisor_at_32(BASE_SEED_32, offset + i);
                cpu_chunk[i] = (u32)(N % cpp_int(p));
            }
        } else {
            for (int i = 0; i < chunk; ++i) {
                cpu_chunk[i] = (u32)(N % divisors[offset + i]);
            }
        }
        results.total_cpu_ms += ms_since(t_cpu_start);
        
        // Verify
        for (int i = 0; i < chunk; ++i) {
            if (cpu_chunk[i] != gpu_chunk[i]) {
                if (results.total_mism < 5) {
                    fprintf(stderr, "Mismatch at %d: cpu=%u gpu=%u\n",
                           offset + i, cpu_chunk[i], gpu_chunk[i]);
                }
                ++results.total_mism;
            }
        }
    }
    
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_out));
}

void run_crt_benchmark_64(int M, const cpp_int& N, 
                         const std::vector<cpp_int>& divisors,
                         const DeviceCRTData& crt_data, int k_used,
                         BenchmarkResults& results,
                         const u64 BASE_SEED_64, const u32 BASE_SEED_32) {
    const int CHUNK_SIZE = 10000000;
    
    // Allocate device buffers
    u64 *d_P = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_P, M * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&d_out, M * sizeof(u64)));
    
    if (GPU_GENERATE_DIVISORS) {
        int threads = 256;
        int blocks = (M + threads - 1) / threads;
        // CRITICAL: Use same offset as CPU (100M) to avoid CRT prime collisions
        const int START_OFFSET = 100000000;
        u64 offset_seed = BASE_SEED_64 + (u64)START_OFFSET * 104729ull;
        generate_divisors_kernel_64<<<blocks, threads>>>(offset_seed, M, d_P);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        std::vector<u64> P_cpu(M);
        for (int i = 0; i < M; ++i) {
            P_cpu[i] = (u64)divisors[i];
        }
        CUDA_CHECK(cudaMemcpy(d_P, P_cpu.data(), M * sizeof(u64), cudaMemcpyHostToDevice));
    }
    
    std::vector<u64> gpu_chunk, cpu_chunk;
    
    for (int offset = 0; offset < M; offset += CHUNK_SIZE) {
        int chunk = std::min(CHUNK_SIZE, M - offset);
        u64 *d_P_chunk = d_P + offset;
        u64 *d_out_chunk = d_out + offset;
        
        // Kernel timing
        int threads = 256;
        int blocks = (chunk + threads - 1) / threads;
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        
        remainders_via_crt_64<<<blocks, threads>>>(
            d_P_chunk, d_out_chunk, chunk, crt_data.d_c, crt_data.d_m, k_used
        );
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        
        float gpu_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t0, t1));
        results.kernel_ms += gpu_ms;
        results.total_gpu_ms += gpu_ms;
        
        // D2H timing
        auto t_d2h_start = now_tp();
        gpu_chunk.resize(chunk);
        CUDA_CHECK(cudaMemcpy(gpu_chunk.data(), d_out_chunk, 
                             chunk * sizeof(u64), cudaMemcpyDeviceToHost));
        double d2h_time = ms_since(t_d2h_start);
        results.d2h_chunks_ms += d2h_time;
        results.total_gpu_ms += d2h_time;
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
        
        // CPU compute
        auto t_cpu_start = now_tp();
        cpu_chunk.resize(chunk);
        if (GPU_GENERATE_DIVISORS) {
            for (int i = 0; i < chunk; ++i) {
                u64 p = divisor_at_64(BASE_SEED_64, offset + i);
                cpu_chunk[i] = (u64)(N % cpp_int(p));
            }
        } else {
            for (int i = 0; i < chunk; ++i) {
                cpu_chunk[i] = (u64)(N % divisors[offset + i]);
            }
        }
        results.total_cpu_ms += ms_since(t_cpu_start);
        
        // Verify
        for (int i = 0; i < chunk; ++i) {
            if (cpu_chunk[i] != gpu_chunk[i]) {
                if (results.total_mism < 5) {
                    fprintf(stderr, "Mismatch at %d: cpu=%llu gpu=%llu\n",
                           offset + i, 
                           (unsigned long long)cpu_chunk[i], 
                           (unsigned long long)gpu_chunk[i]);
                }
                ++results.total_mism;
            }
        }
    }
    
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_out));
}

void print_benchmark_results(int M, const BenchmarkResults& results, int k_used, int k) {
    double gpu_mps = (results.total_gpu_ms > 0) ? 
                     (M / 1e6) / (results.total_gpu_ms / 1000.0) : 0.0;
    double cpu_mps = (results.total_cpu_ms > 0) ? 
                     (M / 1e6) / (results.total_cpu_ms / 1000.0) : 0.0;
    
    double ms_per_cand_gpu = (results.total_gpu_ms > 0) ? 
                             results.total_gpu_ms / double(M) :
                             std::numeric_limits<double>::quiet_NaN();
    double ms_per_cand_cpu = (results.total_cpu_ms > 0) ? 
                             results.total_cpu_ms / double(M) :
                             std::numeric_limits<double>::quiet_NaN();
    double Mcross = (ms_per_cand_gpu > 0) ? 
                    (ms_per_cand_cpu / ms_per_cand_gpu) : -1.0;
    
    printf("\n=== Results for M=%d (%d-bit divisors) ===\n", M, DIVISOR_BITS);
    printf("M=%d: ms per candidate: GPU=%.9f, CPU=%.9f | CPU/GPU crossover @ M=%.9f\n",
           M, ms_per_cand_gpu, ms_per_cand_cpu, Mcross);
    printf("[profile] runtime: h2d_chunks=%.2fms kernel=%.2fms d2h_chunks=%.2fms | external: pgen=%.2fms\n",
           results.h2d_chunks_ms, results.kernel_ms, results.d2h_chunks_ms, results.pgen_ms);
    
    if (DIVISOR_BITS <= 64) {
        printf("k=%d%s candidates=%d | CRT-GPU %.3f ms (%.2f M/s) | "
               "CPU %.3f ms (%.2f M/s) | mismatches=%d\n",
               k_used, (k < 0 ? " (dyn)" : ""), M,
               results.total_gpu_ms, gpu_mps, results.total_cpu_ms, cpu_mps, 
               results.total_mism);
        printf("CRT Kernel-only: %.2f M/s\n", gpu_mps);
    }
}

#ifndef NO_CGBN
void run_cgbn_benchmark(int M, const cpp_int& N, 
                       const std::vector<cpp_int>& divisors,
                       cgbn_bn_mem_t* d_N_single) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t required_bytes = M * sizeof(cgbn_bn_mem_t) * 3;
    
    if (required_bytes > free_mem * 0.8) {
        printf("[CGBN div_rem] Skipping: M=%d requires ~%.2f GB, only %.2f GB free.\n",
               M, required_bytes / 1e9, free_mem / 1e9);
        return;
    }
    
    size_t nbits = bitlen_cppint(N);
    size_t max_div_bits = 0;
    for (int i = 0; i < M; ++i) {
        size_t db = bitlen_cppint(divisors[i]);
        if (db > max_div_bits) max_div_bits = db;
    }
    
    if (nbits > CGBN_BITS || max_div_bits > CGBN_BITS) {
        size_t needed = std::max(nbits, max_div_bits);
        fprintf(stderr, "[CGBN div_rem] Skipping: Need %zu bits, CGBN_BITS=%d. "
                "Rebuild with -DCGBN_BITS=%zu\n", 
                needed, CGBN_BITS, ((needed + 31) / 32) * 32);
        return;
    }
    
    printf("[CGBN div_rem] Running comparison benchmark\n");
    
    // Allocate pinned host memory
    cgbn_bn_mem_t *h_divs = nullptr, *h_r = nullptr;
    cudaHostAlloc(&h_divs, M * sizeof(cgbn_bn_mem_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_r, M * sizeof(cgbn_bn_mem_t), cudaHostAllocDefault);
    
    for (int i = 0; i < M; ++i) {
        cppint_to_cgbn_mem(divisors[i], h_divs[i]);
    }
    
    // Allocate device memory
    cgbn_bn_mem_t *d_divs = nullptr, *d_qm = nullptr, *d_rm = nullptr;
    cgbn_error_report_t *report = nullptr;
    
#ifdef CGBN_ALLOCATE_ERROR_REPORT
    report = CGBN_ALLOCATE_ERROR_REPORT();
#else
    CUDA_CHECK(cudaMallocManaged(&report, sizeof(cgbn_error_report_t)));
    host_cgbn_error_report_init(report);
#endif
    
    CUDA_CHECK(cudaMalloc(&d_divs, M * sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMalloc(&d_qm, M * sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMalloc(&d_rm, M * sizeof(cgbn_bn_mem_t)));
    
    // H2D timing
    auto t_h2d_start = now_tp();
    CUDA_CHECK(cudaMemcpy(d_divs, h_divs, M * sizeof(cgbn_bn_mem_t),
                         cudaMemcpyHostToDevice));
    double cgbn_h2d_ms = ms_since(t_h2d_start);
    
    // Kernel timing
    cudaEvent_t c0, c1;
    CUDA_CHECK(cudaEventCreate(&c0));
    CUDA_CHECK(cudaEventCreate(&c1));
    
    auto t_launch_start = now_tp();
    CUDA_CHECK(cudaEventRecord(c0));
    
    int threads = 128;
    if (threads % CGBN_TPI) threads += (CGBN_TPI - (threads % CGBN_TPI));
    int blocks = (M * CGBN_TPI + threads - 1) / threads;
    
    cgbn_divrem_kernel<<<blocks, threads>>>(report, d_N_single, d_divs, 
                                           d_qm, d_rm, M);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaEventRecord(c1));
    CUDA_CHECK(cudaEventSynchronize(c1));
    
    double cgbn_launch_ms = ms_since(t_launch_start);
    float cgbn_divrem_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&cgbn_divrem_ms, c0, c1));
    
    // D2H timing
    auto t_d2h_start = now_tp();
    CUDA_CHECK(cudaMemcpy(h_r, d_rm, M * sizeof(cgbn_bn_mem_t),
                         cudaMemcpyDeviceToHost));
    double cgbn_d2h_ms = ms_since(t_d2h_start);
    
    // Verify
    int mism_cgbn = 0;
    for (int i = 0; i < M; ++i) {
        cpp_int cgbn_rem = cgbn_mem_to_cppint(h_r[i]);
        cpp_int expect = N % divisors[i];
        if (cgbn_rem != expect) {
            if (mism_cgbn < 5) {
                fprintf(stderr, "[CGBN div_rem] Mismatch at %d: "
                        "cpu=%s cgbn=%s\n", 
                        i, expect.str().c_str(), cgbn_rem.str().c_str());
            }
            ++mism_cgbn;
        }
    }
    
    // Print results
    double cgbn_mps = (cgbn_divrem_ms > 0) ? 
                     (M / 1e6) / (cgbn_divrem_ms / 1000.0) : 0.0;
    printf("CGBN div_rem | %d divisions | %.3f ms (%.2f M/s) | mismatches=%d\n", 
           M, cgbn_divrem_ms, cgbn_mps, mism_cgbn);
    
    double cgbn_full_ms = cgbn_h2d_ms + (double)cgbn_divrem_ms + cgbn_d2h_ms;
    double cgbn_full_mps = (cgbn_full_ms > 0.0) ? 
                          ((M / 1e6) / (cgbn_full_ms / 1000.0)) : 0.0;
    printf("CGBN div_rem full | %d divisions | %.3f ms (%.2f M/s) | "
           "h2d=%.2f ms kernel=%.2f ms d2h=%.2f ms\n",
           M, cgbn_full_ms, cgbn_full_mps, cgbn_h2d_ms, 
           (double)cgbn_divrem_ms, cgbn_d2h_ms);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_divs));
    CUDA_CHECK(cudaFree(d_qm));
    CUDA_CHECK(cudaFree(d_rm));
    
#ifdef CGBN_FREE_ERROR_REPORT
    CGBN_FREE_ERROR_REPORT(report);
#else
    CUDA_CHECK(cudaFree(report));
#endif
    
    CUDA_CHECK(cudaEventDestroy(c0));
    CUDA_CHECK(cudaEventDestroy(c1));
    
    cudaFreeHost(h_divs);
    cudaFreeHost(h_r);
}
#endif

int main(int argc, char** argv) {
    check_gpu_capability();
    printf("=== CRT GPU with %d-bit divisors ===\n", DIVISOR_BITS);
    
    warmup_gpu_context();
    
    const u64 BASE_SEED_64 = 1234567ull;
    const u32 BASE_SEED_32 = (u32)BASE_SEED_64;
    
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Read N
    std::string Ns;
    std::ifstream fin(argv[1]);
    if (fin) {
        std::getline(fin, Ns);
        fin.close();
    } else {
        Ns = argv[1];
    }
    
    // Parse command line
    int k = -1;
    int argi = 2;
    
    if (argc >= 5 && std::string(argv[2]) == std::string("-k")) {
        char* endptr = nullptr;
        long kval = std::strtol(argv[3], &endptr, 10);
        if (*endptr != '\0' || kval <= 0) {
            fprintf(stderr, "Error: invalid K for -k (got '%s').\n", argv[3]);
            return 1;
        }
        k = (int)kval;
        argi = 4;
    }
    
    if (argc - argi < 1) {
        fprintf(stderr, "Error: At least one M value required.\n");
        return 1;
    }
    
    // Collect M list
    std::vector<int> M_list;
    for (int i = argi; i < argc; ++i) {
        int Mi = std::stoi(argv[i]);
        if (Mi <= 0) {
            fprintf(stderr, "Error: M values must be positive (got %d).\n", Mi);
            return 1;
        }
        M_list.push_back(Mi);
    }
    
    cpp_int N = read_big_decimal(Ns);
    const int SAFETY_BITS = 32;
    
    // Perform CRT setup
    SetupData setup = perform_crt_setup(N, k, SAFETY_BITS);
    
    // Upload CRT data to device
    DeviceCRTData crt_data;
    crt_data.allocate_and_upload(setup.c32, setup.m, setup.k_used);
    
    double total_setup_ms = setup.choose_ms + setup.residues_ms + 
                           setup.garner_ms + crt_data.h2d_ms;
    
    printf("\n=== One-time Setup ===\n");
    printf("choose=%.2fms residues=%.2fms garner=%.2fms h2d_static=%.2fms | total=%.2fms\n",
           setup.choose_ms, setup.residues_ms, setup.garner_ms, 
           crt_data.h2d_ms, total_setup_ms);
    printf("\n");
    if (DIVISOR_BITS == 32) {
        warmup_crt_kernel_32(crt_data.d_c, crt_data.d_m, setup.k_used);
    } else if (DIVISOR_BITS == 64) {
        warmup_crt_kernel_64(crt_data.d_c, crt_data.d_m, setup.k_used);
    }
    
#ifndef NO_CGBN
    cgbn_bn_mem_t N_mem;
    cppint_to_cgbn_mem(N, N_mem);
    cgbn_bn_mem_t *d_N_single = nullptr;
    CUDA_CHECK(cudaMalloc(&d_N_single, sizeof(cgbn_bn_mem_t)));
    CUDA_CHECK(cudaMemcpy(d_N_single, &N_mem, sizeof(cgbn_bn_mem_t), 
                         cudaMemcpyHostToDevice));
#endif
    
    std::unordered_set<u32> mset(setup.m.begin(), setup.m.end());
    double total_runtime_ms = 0.0;
    long long total_divs = 0;
    
    // Process each M
    for (size_t m_idx = 0; m_idx < M_list.size(); ++m_idx) {
        int M = M_list[m_idx];
        BenchmarkResults results;
        
        auto t_pgen_start = now_tp();
        std::vector<cpp_int> divisors = generate_divisors(M, DIVISOR_BITS, mset);
        if ((int)divisors.size() < M) {
            fprintf(stderr, "Warning: Only generated %zu divisors, clamping M to %zu\n", 
                    divisors.size(), divisors.size());
            M = (int)divisors.size();
        }
        results.pgen_ms = ms_since(t_pgen_start);
        
        if (DIVISOR_BITS == 32) {
            run_crt_benchmark_32(M, N, divisors, crt_data, setup.k_used, 
                                results, BASE_SEED_64, BASE_SEED_32);
        } else if (DIVISOR_BITS == 64) {
            run_crt_benchmark_64(M, N, divisors, crt_data, setup.k_used, 
                                results, BASE_SEED_64, BASE_SEED_32);
        } else {
#ifndef NO_CGBN
            fprintf(stderr, "Error: CGBN-CRT kernel is not available in this build.\n");
            continue;
#else
            fprintf(stderr, "Error: DIVISOR_BITS=%d requires CGBN, but NO_CGBN is defined\n", 
                    DIVISOR_BITS);
            continue;
#endif
        }
        
        print_benchmark_results(M, results, setup.k_used, k);
        
        double perM_runtime_ms = results.h2d_chunks_ms + results.kernel_ms + 
                                results.d2h_chunks_ms;
        double amortized_setup_ms = (total_setup_ms * 
                                    (double(M) / double(total_divs + M)));
        double crt_full_ms = amortized_setup_ms + perM_runtime_ms;
        double crt_full_mps = (M / 1e6) / (crt_full_ms / 1000.0);
        printf("CRT Full (amortized): %.2f M/s\n", crt_full_mps);
        
        total_runtime_ms += perM_runtime_ms;
        total_divs += M;
        
#ifndef NO_CGBN
            run_cgbn_benchmark(M, N, divisors, d_N_single);
#endif
    }
    
    // Print global summary
    double crt_full_ms_total = total_setup_ms + total_runtime_ms;
    double crt_full_Mps_amortized = (total_divs / 1e6) / (crt_full_ms_total / 1000.0);
    
    printf("\n=== Amortized CRT Full Throughput Across All M ===\n");
    printf("Total setup: %.2f ms | Total runtime: %.2f ms | Total divisions: %lld\n",
           total_setup_ms, total_runtime_ms, total_divs);
    printf("Amortized CRT-Full Throughput: %.2f M/s\n", crt_full_Mps_amortized);
    
    // Cleanup
    crt_data.free();
#ifndef NO_CGBN
    if (d_N_single) CUDA_CHECK(cudaFree(d_N_single));
#endif
    
    return 0;
}

// CGBN kernel implementation (must be in only ONE .cu file to avoid multiple definitions)
#ifndef NO_CGBN
__global__ void cgbn_divrem_kernel(cgbn_error_report_t *report,
                                   const cgbn_bn_mem_t *d_N_single,
                                   cgbn_bn_mem_t *divs,
                                   cgbn_bn_mem_t *qouts,
                                   cgbn_bn_mem_t *routs,
                                   int count)
{
    cgbn_context ctx(cgbn_no_checks, report, 0);
    cgbn_env env(ctx);
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / CGBN_TPI;
    if (instance >= count) return;
    
    cgbn_bn_t n, d, q, r;
    cgbn_load(env, n, (cgbn_bn_mem_t*)d_N_single);
    cgbn_load(env, d, &divs[instance]);
    
    if (cgbn_compare_ui32(env, d, 0) == 0) {
        cgbn_set_ui32(env, q, 0);
        cgbn_set_ui32(env, r, 0);
    } else {
        cgbn_div_rem(env, q, r, n, d);
    }
    
    cgbn_store(env, &qouts[instance], q);
    cgbn_store(env, &routs[instance], r);
}
#endif