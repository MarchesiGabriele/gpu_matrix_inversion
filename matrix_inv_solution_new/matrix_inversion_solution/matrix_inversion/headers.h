#pragma once
#include <vector>
#include "res_struct.h"

double matrix_multiply(std::vector<double> matriceA, std::vector<double> matriceB);

Res matrix_inversion(std::vector<double> matrix_vector, int matrix_order);

Res matrix_inversion_bench(std::vector<double> matrix_vector, int matrix_order);

std::vector<double> matrix_inversion_no_pivots(std::vector<double> matrix_vector, int matrix_order);

std::vector<double> matrix_inversion_no_pivots_bench(std::vector<double> matrix_vector, int matrix_order);

