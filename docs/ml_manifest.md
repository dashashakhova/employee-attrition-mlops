# ML-манифест проекта Employee Attrition MLOps Platform

## 1. Предыстория

Компания сталкивается с текучестью кадров и хочет заранее выявлять сотрудников с повышенным риском увольнения. Решение должно поддерживать полный ML lifecycle: подготовку данных, обучение, оценку, регистрацию, эксплуатацию и переобучение.

## 2. Ценностное предложение

Сервис оценивает вероятность увольнения сотрудника по HR-признакам. Результат предназначен как аналитический сигнал для HR и не заменяет решение специалиста.

## 3. Цели

1. Подготовка воспроизводимого набора признаков.
2. Обучение и оценка модели бинарной классификации.
3. Хранение экспериментов и версий моделей.
4. Онлайн-сервинг через API.
5. Мониторинг технических и модельных метрик.
6. Автоматический запуск переобучения по событию drift или деградации качества.
7. Quality gate и переключение Production на новую модель только после проверки.

## 4. Решение

Заявленный уровень зрелости: **Level 2**.

Компоненты:
- GitHub — версионирование.
- GitHub Actions — CI/CD.
- PostgreSQL — изменяемый production data source.
- Feature Store layer — подготовка и хранение актуального feature dataset.
- Airflow — оркестрация.
- MLflow — experiment tracking и model registry.
- FastAPI — online inference.
- Prometheus client — метрики API.
- Drift detection — статистический контроль данных.
- Quality gate — блокировка плохой модели.
- Production pointer — переключение API на новую версию.
- Docker Compose и Render Blueprint — декларативная инфраструктура.

Жизненный цикл:

Production batch → drift/quality event → Feature Store → training → MLflow → quality gate → promotion → API.

## 5. Осуществимость

Проект рассчитан на одного ML/MLOps-разработчика. Используются открытые инструменты и Docker.

## 6. Данные

### Обучающие данные

IBM HR Analytics Employee Attrition Dataset хранится в `data/raw/hr_attrition.csv`.

### Production data

Реальная корпоративная HR-система в рамках учебного проекта недоступна. Поэтому создан PostgreSQL-backed production data simulator.

Первичная загрузка формирует reference batch из исторического датасета. Последующие запуски ingestion добавляют новые batch'и с timestamp и batch ID. В учебной демонстрации часть новых записей намеренно изменяется: увеличивается доля `OverTime = Yes` и сдвигается `MonthlyIncome`. Это позволяет воспроизводимо создать data drift.

Если доступна реальная HR DB, DWH или streaming source, simulator заменяется ingestion-коннектором; downstream pipeline сохраняет тот же контракт.

Разметка: `Attrition = Yes → 1`, `Attrition = No → 0`.

## 7. Метрики

Бизнес: динамика attrition rate относительно baseline.

ML: ROC-AUC, Precision, Recall, F1.

Технические: P95 latency, error rate, availability, request volume.

## 8. Оценка качества

Baseline Logistic Regression:
- ROC-AUC = 0.8034.
- Precision = 0.8095.
- Recall = 0.3617.
- F1 = 0.5000.

Quality gate: ROC-AUC ≥ 0.80 и Recall ≥ 0.36.

## 9. Подбор модели

Первая итерация — Logistic Regression. Следующие кандидаты сравниваются в MLflow по одинаковому протоколу train/test.

## 10. Инференс

Real-time REST API: `/health`, `/predict`, `/metrics`.

## 11. Обратная связь

Новый production batch анализируется на data drift. Для числовых признаков используется Kolmogorov–Smirnov test с α = 0.05, для категориальных — TVD с порогом 0.10.

Переобучение запускается, если обнаружен drift либо метрики последнего production batch не проходят quality threshold.

## 12. Управление проектом

Команда: 1 ML/MLOps Engineer.

Артефакты: код, Docker infrastructure, Airflow DAG, Feature Store layer, MLflow registry, API, monitoring, CI/CD и документация.

## 13. Ограничения

Production data simulator используется только для воспроизводимой демонстрации изменения данных. Это явно отличается от реального корпоративного источника и не выдается за него.