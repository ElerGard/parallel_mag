#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <iomanip>

#define cols 4096
#define rows 4096

void add_matrix(double* A, const double* B, const double* C, size_t total_elements)
{
    for (size_t i = 0; i < total_elements; i++)
    {
        A[i] = B[i] + C[i];
    }
}

void add_matrix_avx2(double* C, const double* A, const double* B, size_t total_elements)
{
    for (size_t i = 0; i < total_elements / 4; i++)
    {
        __m256d a = _mm256_loadu_pd(&(A[i * 4]));
        __m256d b = _mm256_loadu_pd(&(B[i * 4]));
        __m256d c = _mm256_add_pd(a, b);
        _mm256_storeu_pd(&(C[i * 4]), c);
    }

    for (size_t i = (total_elements / 4) * 4; i < total_elements; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// Выводим только 5 строк и столбцов
void print_matrix(const double* matrix, size_t colsc, size_t rowsc, size_t max_rows = 5, size_t max_cols = 5)
{
    std::cout << std::fixed << std::setprecision(2);
    for (size_t r = 0; r < std::min(rowsc, max_rows); ++r)
    {
        for (size_t c = 0; c < std::min(colsc, max_cols); ++c)
        {
            std::cout << matrix[r * colsc + c] << " ";
        }
        std::cout << (colsc > max_cols ? "... " : "") << "\n";
    }
    if (rowsc > max_rows)
        std::cout << "... \n";
    std::cout << "\n";
}

int main()
{
    size_t total_elements = cols * rows;
    std::vector<double> B(total_elements, 1), C(total_elements, -1), A(total_elements);

    auto calculate_average_time = [](auto func, double* A, const double* B, const double* C, size_t total_elements, int iterations) {
        double total_time = 0;
        for (int i = 0; i < iterations; i++)
        {
            auto t1 = std::chrono::steady_clock::now();
            func(A, B, C, total_elements);
            auto t2 = std::chrono::steady_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }
        return total_time / iterations;
    };

    const int num_attempts = 10;

    std::cout << "Матрица B:\n";
    print_matrix(B.data(), cols, rows);

    std::cout << "Матрица С:\n";
    print_matrix(C.data(), cols, rows);

    double avg_time_basic = calculate_average_time(add_matrix, A.data(), B.data(), C.data(), total_elements, num_attempts);
    std::cout << "Среднее время. Обычное сложение: " << avg_time_basic << " мс.\n";

    std::cout << "Результат обчыного сложения:\n";
    print_matrix(A.data(), cols, rows);

    std::fill_n(A.data(), total_elements, 0);
    std::fill_n(B.data(), total_elements, 1);
    std::fill_n(C.data(), total_elements, -1);

    double avg_time_avx2 = calculate_average_time(add_matrix_avx2, A.data(), B.data(), C.data(), total_elements, num_attempts);
    std::cout << "Среднее вермя. AVX2: " << avg_time_avx2 << " мс.\n";

    std::cout << "Результать векторного сложения\n";
    print_matrix(A.data(), cols, rows);

    return 0;
}
