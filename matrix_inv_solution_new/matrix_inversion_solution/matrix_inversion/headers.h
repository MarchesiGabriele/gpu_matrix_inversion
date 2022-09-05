#pragma once
#include <vector>

double matrix_multiply(std::vector<double> matriceA, std::vector<double> matriceB);

std::vector<double> matrix_inversion(std::vector<double> matrix_vector, int matrix_order);

std::vector<double> matrix_inversion_no_pivots(std::vector<double> matrix_vector, int matrix_order);

std::vector<double> matrix_inversion_improved(std::vector<double> matrix_vector, int matrix_order);

void pivot_max_test(std::vector<double> matrix_vector, int matrix_order);
