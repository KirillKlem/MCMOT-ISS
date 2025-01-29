# MCMOT: Многокамерный Многоперсональный Трекер Объектов 🚶‍♂️🎥

[🔗 Эксперимент на 1 камере](https://drive.google.com/drive/folders/1HEW4QMCBh-hO_yZ-gFm9FxI3i1Pnt3En?usp=sharing)  
[🔗 Эксперимент на 2 камерах](https://drive.google.com/drive/folders/1Bu0eiNzoGKboak8udsyDJYeNiZ-MsCEw?usp=sharing)  
[🔗 Эксперимент на 5 камерах](https://drive.google.com/drive/folders/1CY83bJwhCwnkGEUvU5oaH_1FZ8hpmIqf?usp=sharing)  
[🔗 DataDump с источниками](https://docs.google.com/document/d/1pzy4N5eHGiqp4lH3r4ggsgNnijfuKgvlNlmZQEfMseI/edit?tab=t.0)

---

## Модели

[📂 Скачать модели reID](https://drive.google.com/drive/folders/1HhoybORgO1O5dPV7AlhdCZKXMWB9ukTj?usp=sharing)  
*(Загрузите файлы в папку `assets` для корректной работы.)*

[📂 Скачать любую из обученных нами моделей](https://drive.google.com/drive/folders/1FUs-KjeIzeJwQNuuX71a9HEF-WYZLhml?usp=drive_link)  

---

## 📜 О проекте

Добро пожаловать в **MCMOT** — передовой многокамерный трекер объектов, разработанный для надежного и точного отслеживания нескольких людей одновременно в различных камерах. Проект объединяет передовые технологии, такие как YOLO для детекции и Metric Learning для ReID, чтобы обеспечить высокую точность и стабильность идентификации.

---

## 🌟 Ключевые особенности

### 1️⃣ Многокамерный трекинг
- Поддержка отслеживания объектов в нескольких камерах с сохранением уникальных идентификаторов.

### 2️⃣ Гибкий выбор модели
- Возможность использовать различные предобученные модели YOLO для оптимального соответствия вашим требованиям:
  - **`yolo11m_trained.pt`**: Средняя точность и скорость.
  - **`yolo11n_trained.pt`**: Высокая скорость с небольшим снижением точности.
  - **`yolo11s_trained.pt`**: Оптимизированная модель для быстродействия.

### 3️⃣ Передовая детекция
- Fine-tuning YOLOv11 для точного распознавания объектов в сложных условиях.

### 4️⃣ ReID модель
- Использование Metric Learning для сопоставления людей между кадрами и камерами.

### 5️⃣ Визуализация и логирование
- Автоматическое наложение аннотаций на видео с ID и Confidence, а также подробное логирование.

### 6️⃣ Простота и масштабируемость
- Легкая интеграция в существующие системы и поддержка нескольких камер без потери производительности.




## 🎯 Преимущества проекта

- **Высокая точность:** Современные алгоритмы обеспечивают минимальные ошибки трекинга.  
- **Масштабируемость:** Легко адаптируется для работы с большим количеством камер.  
- **Гибкость:** Поддержка выбора моделей позволяет оптимизировать систему под задачи.  
- **Адаптивность:** Система автоматически подстраивается под изменения в окружающей среде.  
- **Интуитивное управление:** Логика кодовой базы ясна даже для начинающих разработчиков.

---

## 🏆 Достижения

- **Эффективный трекинг:** Надежная система, обеспечивающая согласованность идентификаторов.  
- **Управление галереей эмбеддингов:** Стабильная работа и предотвращение частых изменений идентификаторов.  
- **Интеграция YOLO и ReID:** Успешное объединение двух передовых технологий.  
- **Удобная визуализация и логи:** Разработаны механизмы для удобного анализа трекинга.  

---

## 🚧 Проблемы и Обоснования

Несмотря на достижения, разработка не обошлась без вызовов:

- **Ограничения вычислительной мощности:** Настройка порогов требует больших ресурсов.  
- **Разнообразие условий съемки:** Освещение, углы обзора и другие параметры усложняют калибровку.  
- **Вариативность эмбеддингов:** Изменения поз, одежды и других факторов влияют на стабильность мэтчинга.

*Эти ограничения вдохновили нас на разработку адаптивных решений.*

---

## 🔮 Будущие Улучшения

1. **Адаптивная настройка порогов**  
   Использование динамических методов для улучшения мэтчинга в реальном времени.

2. **Интеграция пространственных ограничений**  
   Применение геометрической информации для улучшения глобального трекинга.

3. **Расширение функционала**  
   Поддержка большего числа камер и улучшение масштабируемости.

4. **Улучшение управления треками**  
   Устойчивость к потерям объектов и более точное предсказание траекторий.

5. **Оптимизация производительности**  
   Повышение скорости обработки видео без компромиссов в точности.

---

Спасибо за ваш интерес к нашему проекту
