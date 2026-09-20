# SLI / SLO и управление рисками

Ниже приведены **целевые SLO** системы. Значения не являются утверждением о фактическом достижении каждого SLO в облаке.

## 1. Технический уровень

| SLI | Что измеряем | SLO | Реакция |
|---|---|---:|---|
| Availability | доля успешных запросов | ≥ 99% | расследование инцидента |
| P95 latency | 95-й перцентиль `/predict` | ≤ 2.0 s | оптимизация/масштабирование |
| Error rate | доля HTTP 5xx | ≤ 1% | расследование |
| Prediction volume | количество запросов | отсутствие необъяснимого нулевого потока | проверка API/клиента |

## 2. Модельный уровень

| SLI | Что измеряем | SLO | Реакция |
|---|---|---:|---|
| ROC-AUC | качество классификации | ≥ 0.80 | блокировка promotion |
| Recall | доля найденных фактических увольнений | ≥ 0.36 | блокировка promotion / retraining |
| Precision | доля корректных positive predictions | ≥ 0.70 | анализ модели |
| F1 | баланс Precision и Recall | ≥ 0.50 | анализ модели |
| Numeric drift | KS-test | p-value ≥ 0.05 | при нарушении — retraining event |
| Categorical drift | TVD | ≤ 0.10 | при нарушении — retraining event |

## 3. Бизнес-уровень

| SLI | Что измеряем | SLO |
|---|---|---:|
| Attrition rate | доля увольнений за период | отслеживать относительно baseline |
| At-risk recall | доля реальных увольнений, заранее попавших в группу риска | ≥ 70% как целевой KPI |
| Coverage | доля сотрудников с рассчитанным риском | ≥ 95% |

## Реакции

Technical SLO breach → проверка API и контейнера.

Model quality breach → текущая Production-модель сохраняется, новая версия не продвигается.

Data drift или production quality degradation → Airflow формирует retraining path.

Успешное обучение → MLflow Registry + quality gate → production pointer переключается на новую модель.