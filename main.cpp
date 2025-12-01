#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <algorithm>
#include <execution>

// універсальна функція для вимірювання часу виконання будь-якої операції
template <typename F>
double measure_ms(F&& f, int reps = 5) {
    using clock = std::chrono::steady_clock;
    std::vector<double> times;
    times.reserve(reps);

    for (int i = 0; i < reps; ++i) {
        auto t0 = clock::now();
        f();
        auto t1 = clock::now();

        std::chrono::duration<double, std::milli> dt = t1 - t0;
        times.push_back(dt.count());
    }

    double sum = 0.0;
    for (double t : times) sum += t;
    return sum / times.size();
}

// власний паралельний none_of: розбиває дані на K частин, кожну обробляє в окремому потоці
// фінальний результат: none_of на всьому діапазоні
template <typename Pred>
bool custom_parallel_none_of(const std::vector<int>& data, Pred pred, int K) {
    size_t N = data.size();
    if (N == 0) return true;  // для пустого діапазону none_of повертає true
    if (K <= 1) {
        return std::none_of(data.begin(), data.end(), pred);
    }

    // захист від занадто великого K
    if (static_cast<size_t>(K) > N) {
        K = static_cast<int>(N);
        if (K == 0) return true;
    }

    std::vector<bool> partial(K, true); // для none_of — нейтральний елемент true (AND)
    std::vector<std::thread> threads;
    threads.reserve(K);

    // використовуємо "стелю": розмір блоку = ceil(N / K)
    size_t chunk = (N + K - 1) / K;

    for (int i = 0; i < K; ++i) {
        size_t start = static_cast<size_t>(i) * chunk;
        if (start >= N) {
            partial[i] = true;
            continue;
        }
        size_t end = std::min(start + chunk, N);

        threads.emplace_back([&, i, start, end]() {
            bool r = std::none_of(data.begin() + start,
                                  data.begin() + end,
                                  pred);
            partial[i] = r;
        });
    }

    for (auto &t : threads) {
        if (t.joinable()) t.join();
    }

    // для none_of потрібно, щоб УСІ піддіапазони не містили елементів, що задовольняють предикат
    bool final_result = std::all_of(partial.begin(), partial.end(),
                                    [](bool v){ return v; });
    return final_result;
}

int main() {
    // 1. Розміри послідовностей, які будемо тестувати
    std::vector<size_t> sizes = {100000, 1000000, 5000000};

    // 2. Значення K для власного паралельного алгоритму
    std::vector<int> K_values = {1, 2, 4, 8, 16, 32};

    // 3. Генератор випадкових чисел
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<int> dist(0, 1'000'000);

    // кількість апаратних потоків
    unsigned hw_threads = std::thread::hardware_concurrency();
    std::cout << "Апаратних потоків CPU: " << hw_threads << "\n\n";

    for (size_t N : sizes) {

        std::cout << "=== N = " << N << " ===\n";

        // --- (A) Генерація даних ---
        std::vector<int> data(N);
        for (auto &x : data) {
            x = dist(rng);
        }

        int threshold = 2'000'000;

        // --- (B) Легкий предикат ---
        auto light_pred = [threshold](int x) {
            // Дуже проста операція: одне порівняння
            return x > threshold;   // завжди false для нашого набору
        };

        // --- (C) Важкий предикат ---
        auto heavy_pred = [threshold](int x) {
            // Імітація важких обчислень
            double s = 0.0;
            for (int i = 0; i < 100; ++i) {
                s += (x * 0.000001 + i) / (i + 1.0);
            }
            // Умова завідомо false — щоб проходити всю послідовність
            return (x > threshold) && (s > 1e9);
        };

        // --- 1) std::none_of без політики ---
        {
            double t_light = measure_ms([&]() {
                std::none_of(data.begin(), data.end(), light_pred);
            });

            double t_heavy = measure_ms([&]() {
                std::none_of(data.begin(), data.end(), heavy_pred);
            });

            std::cout << "std::none_of (без політики): "
                      << "легкий = " << t_light << " ms, "
                      << "важкий = " << t_heavy << " ms\n";
        }

        // --- 2) std::none_of з політиками ---
        {
            double t_seq_light = measure_ms([&]() {
                std::none_of(std::execution::seq, data.begin(), data.end(), light_pred);
            });
            double t_seq_heavy = measure_ms([&]() {
                std::none_of(std::execution::seq, data.begin(), data.end(), heavy_pred);
            });

            double t_par_light = measure_ms([&]() {
                std::none_of(std::execution::par, data.begin(), data.end(), light_pred);
            });
            double t_par_heavy = measure_ms([&]() {
                std::none_of(std::execution::par, data.begin(), data.end(), heavy_pred);
            });

            double t_par_unseq_light = measure_ms([&]() {
                std::none_of(std::execution::par_unseq, data.begin(), data.end(), light_pred);
            });
            double t_par_unseq_heavy = measure_ms([&]() {
                std::none_of(std::execution::par_unseq, data.begin(), data.end(), heavy_pred);
            });

            std::cout << "std::none_of (seq):       "
                      << "легкий = " << t_seq_light << " ms, "
                      << "важкий = " << t_seq_heavy << " ms\n";

            std::cout << "std::none_of (par):       "
                      << "легкий = " << t_par_light << " ms, "
                      << "важкий = " << t_par_heavy << " ms\n";

            std::cout << "std::none_of (par_unseq): "
                      << "легкий = " << t_par_unseq_light << " ms, "
                      << "важкий = " << t_par_unseq_heavy << " ms\n";
        }

        // --- 3) Власний паралельний none_of: таблиця по K ---
        std::cout << "\nK, t_custom_light_ms, t_custom_heavy_ms\n";

        double best_light_time = 1e300;
        int    best_light_K    = -1;

        double best_heavy_time = 1e300;
        int    best_heavy_K    = -1;

        for (int K : K_values) {
            if (K <= 0) continue;

            double t_custom_light = measure_ms([&]() {
                custom_parallel_none_of(data, light_pred, K);
            });

            double t_custom_heavy = measure_ms([&]() {
                custom_parallel_none_of(data, heavy_pred, K);
            });

            std::cout << K << ", "
                      << t_custom_light << ", "
                      << t_custom_heavy << "\n";

            if (t_custom_light < best_light_time) {
                best_light_time = t_custom_light;
                best_light_K    = K;
            }
            if (t_custom_heavy < best_heavy_time) {
                best_heavy_time = t_custom_heavy;
                best_heavy_K    = K;
            }
        }

        // Вивід найкращого K та співвідношення з кількістю потоків
        if (best_light_K != -1) {
            std::cout << "\nНайкраще K для легкого предиката: " << best_light_K
                      << " (час = " << best_light_time << " ms)";
            if (hw_threads > 0) {
                std::cout << ", K / hw_threads = "
                          << static_cast<double>(best_light_K) / hw_threads;
            }
            std::cout << "\n";
        }

        if (best_heavy_K != -1) {
            std::cout << "Найкраще K для важкого предиката: " << best_heavy_K
                      << " (час = " << best_heavy_time << " ms)";
            if (hw_threads > 0) {
                std::cout << ", K / hw_threads = "
                          << static_cast<double>(best_heavy_K) / hw_threads;
            }
            std::cout << "\n";
        }

        std::cout << "\n----------------------------------------\n\n";
    }

    return 0;
}

