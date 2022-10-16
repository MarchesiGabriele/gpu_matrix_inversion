#pragma once
#include <vector>
#include "res_struct.h"

double matrix_multiply(std::vector<double> matriceA, std::vector<double> matriceB);

std::vector<float> matrix_inversion_FP32(std::vector<float> matrix_vector, int matrix_order);

std::vector<double> matrix_inversion_FP64(std::vector<double> matrix_vector, int matrix_order);

std::vector<double> matrix_inversion_no_pivots(std::vector<double> matrix_vector, int matrix_order);

// BENCHMARKS
Res no_pivots_bench(std::vector<double> matrix_vector, int matrix_order);
Res FP32_bench(std::vector<float> matrix_vector, int matrix_order);
Res FP64_bench(std::vector<double> matrix_vector, int matrix_order);
