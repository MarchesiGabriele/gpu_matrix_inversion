#pragma once
#include <vector>
#include "res_struct.h"

void matrix_multiply(std::vector<float> matriceA, std::vector<float> matriceB);

std::vector<float> matrix_inversion_FP32(std::vector<float> matrix_vector, int matrix_order);

std::vector<double> matrix_inversion_FP64(std::vector<double> matrix_vector, int matrix_order);

Res matrix_inversion_bench(std::vector<double> matrix_vector, int matrix_order);

std::vector<double> matrix_inversion_no_pivots(std::vector<double> matrix_vector, int matrix_order);
Res matrix_inversion_no_pivots_bench(std::vector<double> matrix_vector, int matrix_order);

