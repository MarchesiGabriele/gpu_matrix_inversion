#pragma once
#include <vector>

namespace matrix_inversion_project{
	class MatrixInversion {
		public:		
			static std::vector<float> matrix_inversion(std::vector<float> matrix_vector, int matrix_order);
	};
}
