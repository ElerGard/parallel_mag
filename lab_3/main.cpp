#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstring> 
#include <immintrin.h>  
#include <iomanip>

using namespace std;

void mul_matrix(double* A, size_t cA, size_t rA,
                const double* B, size_t cB, size_t rB,
                const double* C, size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < cA; i++)
    {
        for (size_t j = 0; j < rA; j++)
        {
            A[i * rA + j] = 0;
            for (size_t k = 0; k < cB; k++)
            {
                A[i * rA + j] += B[k * rB + j] * C[i * rC + k];
            }
        }
    }
}

void mul_matrix_avx2(double* A, 
                     size_t cA, size_t rA,
                     const double* B, 
                     size_t cB, size_t rB,
                     const double* C,
                     size_t cC, size_t rC)
{
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < rB / 4; i++)
    {
        for (size_t j = 0; j < cC; j++)
        {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < rC; k++)
            {
                __m256d bCol = _mm256_loadu_pd(B + rB * k + i * 4);
                __m256d broadcasted = _mm256_set1_pd(C[j * rC + k]);
                __m256d mul_result = _mm256_mul_pd(bCol, broadcasted); 
                sum = _mm256_add_pd(sum, mul_result); 
            }

            _mm256_storeu_pd(A + j * rA + i * 4, sum);
        }
    }
}

vector<double> generate_permutation_matrix(std::size_t n)
{
    vector<double> permut_matrix(n * n, 0);
    
    for (std::size_t i = 0; i < n; i++)
    {
        permut_matrix[(i + 1) * n - 1 - i] = 1;
    }

    return permut_matrix;
}

void randomize_matrix(double* matrix, std::size_t matrix_order)
{
    std::uniform_real_distribution<double> unif(0, 100000);
    std::default_random_engine re;
    for (std::size_t i = 0; i < matrix_order * matrix_order; i++)
    {
        matrix[i] = unif(re);
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

int main(int argc, char** argv)
{
    const int num_attempts = 10;
    const std::size_t matrix_order = 16 * 4 * 9;

    auto calculate_average_time = [](auto func, double* A, const double* B, const double* C, size_t matrix_order, int iterations) {
        double total_time = 0;
        for (int i = 0; i < iterations; i++)
        {
            randomize_matrix(A, matrix_order);
            auto t1 = std::chrono::steady_clock::now();
            func(A, matrix_order, matrix_order,
                B, matrix_order, matrix_order,
                C, matrix_order, matrix_order);
            auto t2 = std::chrono::steady_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }
        return total_time / iterations;
    };

    vector<double> A(matrix_order * matrix_order),
                   B = generate_permutation_matrix(matrix_order),
                   C(matrix_order * matrix_order),
                   D(matrix_order * matrix_order);

    randomize_matrix(A.data(), matrix_order);

    cout << "Матрица A:\n";
    print_matrix(A.data(), matrix_order, matrix_order);

    cout << "Матрица B:\n";
    print_matrix(B.data(), matrix_order, matrix_order);

    mul_matrix(C.data(), matrix_order, matrix_order,
               A.data(), matrix_order, matrix_order,
               B.data(), matrix_order, matrix_order);

    mul_matrix_avx2(D.data(), matrix_order, matrix_order,
                    A.data(), matrix_order, matrix_order,
                    B.data(), matrix_order, matrix_order);

    if (memcmp(static_cast<void*>(C.data()),
               static_cast<void*>(D.data()),
               matrix_order * matrix_order * sizeof(double)))
    {
        cout << "Результат перемножения некорректен";
        return -1;
    }

    double avg_time_basic = calculate_average_time(mul_matrix, C.data(), A.data(), B.data(), matrix_order, num_attempts);
    cout << "Скалярное умножение: время = " << avg_time_basic << "мс, ускорение = 1\n";


    double avg_time_avx2 = calculate_average_time(mul_matrix_avx2, D.data(), A.data(), B.data(), matrix_order, num_attempts);
    cout << "Векторное умножение: время = " << avg_time_avx2
         << "мс, ускорение = " << avg_time_basic / avg_time_avx2 << "\n";


    return 0;
}
