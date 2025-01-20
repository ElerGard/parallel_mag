#include <iostream>
#include <omp.h>
#include <fstream>
#include <thread>
#include <vector>
#include <numeric>

const size_t N = 100000000; 

double f(double x)
{
    return x * x;
}

double integrate(double a, double b)
{
    double sum = 0;
    double dx = (b - a) / N;

    for (size_t i = 0; i < N; i++)
    {
        sum += f(a + i * dx);
    }

    return dx * sum;
}

double integrate_omp(double a, double b, int threads)
{
    double sum = 0;
    double dx = (b - a) / N;

    omp_set_num_threads(threads);
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; i++)
    {
        sum += f(a + i * dx);
    }

    return dx * sum;
}

int main()
{
    std::ofstream output("reduction.csv");

    output << "Threads,Attempt,Duration,Acceleration\n";

    const int num_attempts = 10;
    double t1 = omp_get_wtime();
    double result = integrate(-1, 1);
    double single_thread_duration = omp_get_wtime() - t1;

    std::cout << "Однопоточный вариант: длительность = " << single_thread_duration << "s\n";

    for (int threads = 2; threads <= std::thread::hardware_concurrency(); threads++)
    {
        std::vector<double> durations;
        durations.reserve(num_attempts);

        for (int attempt = 0; attempt < num_attempts; attempt++)
        {
            t1 = omp_get_wtime();
            result = integrate_omp(-1, 1, threads);
            double duration = omp_get_wtime() - t1;
            durations.push_back(duration);

            double acceleration = single_thread_duration / duration;
            output << threads << "," << (attempt + 1) << "," << duration << "," << acceleration << "\n";
        }
        double average_duration = std::accumulate(durations.begin(), durations.end(), 0.0) / num_attempts;
        double average_acceleration = single_thread_duration / average_duration;

        std::cout << "Потоков: " << threads << ", средняя длительность = " << average_duration << "s, среднее ускорение = " << average_acceleration << "\n";
    }

    output.close();
    return 0;
}
