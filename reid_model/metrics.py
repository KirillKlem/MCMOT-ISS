import numpy as np

# Image retrieval metrics
# Precision@k
def precision_at_k_embeddings(query_embeddings, dataset_embeddings, labels, k=5):
    """
    Вычисляет precision@k, учитывая случаи, когда меток меньше, чем k.
    """
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    dataset_norm = dataset_embeddings / np.linalg.norm(dataset_embeddings, axis=1, keepdims=True)
    similarities = np.dot(query_norm, dataset_norm.T)

    np.fill_diagonal(similarities, -10**6)

    # Get top-k indices for each query
    top_k_indices = np.argpartition(-similarities, range(k), axis=1)[:, :k]

    # Retrieve the top-k labels for each query
    top_k_labels = labels[top_k_indices]

    # Compare query labels to top-k labels and count matches
    correct = (labels[:, None] == top_k_labels).any(axis=1).sum()

    # Calculate precision@k
    precision = correct / len(query_embeddings)
    return precision



#CMC@k (usually k = 1)
def cmc_at_k(query_embeddings, dataset_embeddings, labels, k=1):
    """
    Computes Cumulative Match Characteristic (CMC) at rank k.

    Args:
        query_embeddings (np.ndarray): Embeddings for the query set.
        dataset_embeddings (np.ndarray): Embeddings for the dataset/gallery.
        labels (np.ndarray): Ground truth labels for the dataset/gallery.
        k (int): Rank at which to compute CMC.

    Returns:
        float: CMC@k score.
    """
    # Нормализуем эмбеддинги для ускорения вычислений
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    dataset_embeddings = dataset_embeddings / np.linalg.norm(dataset_embeddings, axis=1, keepdims=True)

    # Вычисляем косинусное сходство (через матричное умножение)
    similarities = np.dot(query_embeddings, dataset_embeddings.T)

    # Исключаем само-сходство
    np.fill_diagonal(similarities, -np.inf)

    # Получаем индексы топ-k
    top_k_indices = np.argpartition(-similarities, k, axis=1)[:, :k]
    top_k_labels = labels[top_k_indices]

    # Сравниваем метки
    matches = np.array([query_label in top_labels for query_label, top_labels in zip(labels, top_k_labels)])
    return matches.mean()