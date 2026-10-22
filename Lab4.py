import math
import random

# ДОПОМІЖНІ МАТЕМАТИЧНІ ФУНКЦІЇ ТА ГЕНЕРАЦІЯ ДАНИХ
def euclidean_distance_sq(p1, p2):
    """Обчислення квадрата евклідової відстані між двома точками."""
    return sum((x - y) ** 2 for x, y in zip(p1, p2))

def generate_sample_data(num_points=100):
    """Генерація двовимірної тестової вибірки (2 штучні кластери + шум)."""
    random.seed(42)
    data = []
    # Перша група точок (навколо [2, 2])
    for _ in range(num_points // 2):
        data.append([random.gauss(2.0, 0.8), random.gauss(2.0, 0.8)])
    # Друга група точок (навколо [7, 7])
    for _ in range(num_points // 2):
        data.append([random.gauss(7.0, 0.8), random.gauss(7.0, 0.8)])
    return data

# 1. МАТРИЧНИЙ МЕТОД: MATRIX PCM (Possibilistic C-Means)
def matrix_pcm(X, num_clusters=2, m=2.0, max_iter=100, tol=1e-5):
    """
    Реалізація методу Possibilistic C-Means (PCM) у матричній формі.
    У PCM знято жорстку умову sum_i(u_ij) = 1 для уникнення впливу викидів.
    """
    n_samples = len(X)
    n_features = len(X[0])
    
    # Випадкова ініціалізація початкових центрів
    random.seed(42)
    initial_indices = random.sample(range(n_samples), num_clusters)
    v = [list(X[idx]) for idx in initial_indices]
    
    # Обчислення масштабувального параметра eta_i для кожного кластера
    # eta_i задає середню дисперсію/радіус кластера
    eta = []
    for i in range(num_clusters):
        d_sum = sum(euclidean_distance_sq(X[j], v[i]) for j in range(n_samples))
        eta.append(d_sum / n_samples)
        
    u = [[0.0] * n_samples for _ in range(num_clusters)]
    
    for iteration in range(max_iter):
        # 1. Оновлення матриці можливостей U
        for i in range(num_clusters):
            for j in range(n_samples):
                d2 = euclidean_distance_sq(X[j], v[i])
                # Формула можливостей PCM: u_ij = 1 / (1 + (d2 / eta_i)^(1 / (m - 1)))
                val = 1.0 + (d2 / max(eta[i], 1e-10)) ** (1.0 / (m - 1.0))
                u[i][j] = 1.0 / val
                
        # 2. Оновлення центрів кластерів V
        v_new = [[0.0] * n_features for _ in range(num_clusters)]
        for i in range(num_clusters):
            num = [0.0] * n_features
            den = 0.0
            for j in range(n_samples):
                weight = u[i][j] ** m
                for k in range(n_features):
                    num[k] += weight * X[j][k]
                den += weight
            
            if den > 0:
                v_new[i] = [num[k] / den for k in range(n_features)]
            else:
                v_new[i] = v[i]
                
        # Перевірка умови зупинки (зміна центрів)
        diff = sum(euclidean_distance_sq(v[i], v_new[i]) for i in range(num_clusters))
        v = v_new
        if diff < tol:
            break
            
    return v, u, iteration + 1

# 2. АДАПТИВНИЙ МЕТОД: ADAPTIVE MATRIX FCM (Adaptive Fuzzy C-Means)
def adaptive_matrix_fcm(X, num_clusters=2, m_start=2.5, m_end=1.2, max_iter=100, tol=1e-5):
    """
    Реалізація Adaptive Matrix FCM.
    Параметр нечіткості m адаптивно зменшується з кожною ітерацією, 
    що прискорює збіжність і стабілізує розподіл на фінальних кроках.
    """
    n_samples = len(X)
    n_features = len(X[0])
    
    # Випадкова ініціалізація матриці належності U з нормуванням за стовпчиками
    random.seed(42)
    u = [[random.random() for _ in range(n_samples)] for _ in range(num_clusters)]
    for j in range(n_samples):
        col_sum = sum(u[i][j] for i in range(num_clusters))
        for i in range(num_clusters):
            u[i][j] /= col_sum
            
    v = [[0.0] * n_features for _ in range(num_clusters)]
    
    for iteration in range(max_iter):
        # Адаптація параметра нечіткості m
        m = m_start - (m_start - m_end) * (iteration / max_iter)
        
        # 1. Обчислення центрів кластерів V
        for i in range(num_clusters):
            num = [0.0] * n_features
            den = 0.0
            for j in range(n_samples):
                weight = u[i][j] ** m
                for k in range(n_features):
                    num[k] += weight * X[j][k]
                den += weight
            v[i] = [num[k] / den for k in range(n_features)]
            
        # 2. Оновлення матриці належності U (з нормуванням FCM)
        u_new = [[0.0] * n_samples for _ in range(num_clusters)]
        for j in range(n_samples):
            distances = [math.sqrt(euclidean_distance_sq(X[j], v[i])) for i in range(num_clusters)]
            for i in range(num_clusters):
                if distances[i] == 0:
                    for c in range(num_clusters):
                        u_new[c][j] = 1.0 if c == i else 0.0
                    break
                else:
                    sum_val = sum((distances[i] / max(distances[c], 1e-10)) ** (2.0 / (m - 1.0)) for c in range(num_clusters))
                    u_new[i][j] = 1.0 / sum_val
                    
        # Перевірка умови зупинки за зміною матриці U
        diff = sum(sum(abs(u_new[i][j] - u[i][j]) for j in range(n_samples)) for i in range(num_clusters))
        u = u_new
        if diff < tol:
            break
            
    return v, u, iteration + 1


# ГОЛОВНА ФУНКЦІЯ ТА ВИВЕДЕННЯ РЕЗУЛЬТАТІВ
def main():
    # 1. Генерація даних
    X = generate_sample_data(num_points=100)
    print(f"Сформовано тестову вибірку: {len(X)} точок у 2D просторі.")
    
    # 2. Виконання Matrix PCM
    v_pcm, u_pcm, iter_pcm = matrix_pcm(X, num_clusters=2)
    print("\n----------------------------------------------------------------------")
    print("1. РЕЗУЛЬТАТИ МАТРИЧНОГО МЕТОДУ (Matrix PCM):")
    print(f"   - Кількість ітерацій до збіжності: {iter_pcm}")
    for i, center in enumerate(v_pcm):
        print(f"   - Центр кластера {i + 1}: X = {center[0]:.4f}, Y = {center[1]:.4f}")
        
    # 3. Виконання Adaptive Matrix FCM
    v_afcm, u_afcm, iter_afcm = adaptive_matrix_fcm(X, num_clusters=2)
    print("\n----------------------------------------------------------------------")
    print("2. РЕЗУЛЬТАТИ АДАПТИВНОГО МЕТОДУ (Adaptive Matrix FCM):")
    print(f"   - Кількість ітерацій до збіжності: {iter_afcm}")
    for i, center in enumerate(v_afcm):
        print(f"   - Центр кластера {i + 1}: X = {center[0]:.4f}, Y = {center[1]:.4f}")
        
    # 4. Порівняння перших 5 точок та їх належності
    print("\n----------------------------------------------------------------------")
    print("3. ПОРІВНЯННЯ СТУПЕНІВ НАЛЕЖНОСТІ (ПЕРШІ 5 ТОЧОК):")
    print("Точка (X, Y)             | PCM U(Cluster 1) | Adaptive FCM U(Cluster 1)")
    print("----------------------------------------------------------------------")
    for j in range(5):
        pt_str = f"[{X[j][0]:.2f}, {X[j][1]:.2f}]"
        print(f"{pt_str:<24} | {u_pcm[0][j]:<16.4f} | {u_afcm[0][j]:<24.4f}")
    print("======================================================================\n")

if __name__ == "__main__":
    main()