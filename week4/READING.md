# Week 4: Monitoring, Drift Detection, and Model Retraining

## The Staleness Problem

A model trained in January works well on January data. But data changes. By February, the statistical properties have shifted. By March, the patterns are different. The model, frozen at January's weights, becomes progressively less accurate.

This is model staleness: the gap between when the model was trained and when it's used grows over time. Unlike traditional software, which is still correct when old, ML models become incorrect when their training distribution no longer matches production.

Model performance degrades over time without retraining—the question is not whether drift happens, but whether you detect it fast enough to respond.

## Data Drift vs Concept Drift

Two types of drift have different signatures and require different responses:

**Data drift (covariate shift)**: The input distribution changes, but the relationship between inputs and outputs remains the same. Example: a model trained on weekday demand sees weekend demand patterns. The distribution of hour_of_day, day_of_week, and trip_count all shift, but the causal relationship is unchanged.

Detection: Compare the distribution of features in recent data to the distribution in training data. Use statistical tests (Kolmogorov-Smirnov test, Population Stability Index, Wasserstein distance) to quantify shift magnitude.

Response: Retraining on recent data usually helps. The model learns the new distribution while preserving the learned relationships.

**Concept drift**: The relationship between inputs and outputs changes. The features look similar, but what they predict has changed. Example: after a pandemic, restaurant demand patterns change fundamentally. The model was trained on pre-pandemic demand; it cannot predict post-pandemic demand even if feature distributions look similar.

Detection: Compare model predictions to ground truth on recent data. Measure accuracy by segment (geography, time of day, customer type). If accuracy drops significantly in specific segments, concept drift is likely.

Response: Retraining helps, but only if you retrain on data that reflects the new relationship. If you keep retraining on data mixed between old and new relationships, the model never learns the new pattern.

Data drift is a distribution problem (solved by retraining); concept drift is a relationship problem (solved by understanding what changed and retraining on new, correct labels).

## Statistical Tests for Drift Detection

Several statistical tests quantify distributional shift:

**Kolmogorov-Smirnov (KS) test**: Measures maximum distance between two cumulative distributions. Returns a p-value: p < 0.05 indicates significant shift. Fast, but only for continuous variables.

**Population Stability Index (PSI)**: Compares expected vs actual proportion in each bin of a feature. PSI < 0.1 is typical; PSI > 0.25 indicates significant shift. Works for categorical and discretized continuous variables.

**Wasserstein distance**: Measures the amount of "work" needed to transform one distribution to another. More flexible than KS test.

**Chi-square test**: For categorical variables, tests if proportions have changed. p < 0.05 indicates shift.

In practice, monitoring systems compute these tests on a rolling basis: compare last week's feature distribution to the previous month's baseline. If tests trigger, alert. Allows humans to review before retraining.

## The Monitoring & Observability Framework

Production ML systems require monitoring at three levels:

**System metrics** (infrastructure): latency (p50, p95, p99), throughput, error rate, resource utilization
- These detect whether the API is running smoothly

**Data metrics** (inputs): null rates, value ranges, feature distributions, cardinality
- These detect data quality issues and drift

**Model metrics** (performance): predictions made, prediction distribution, accuracy on recent data (with ground truth lag), segmented by important subgroups
- These detect concept drift and performance degradation

**Business metrics** (outcome): engagement, revenue, user retention, whatever the system is optimizing for
- These detect whether the model is actually helping

Most ML teams lack monitoring tools for data quality and drift. Teams that do monitor tend to use custom-built solutions or database queries, not purpose-built tools.

Observability (the ability to ask arbitrary questions after failures) is rarer still. Most teams have metrics (known unknowns) but lack observability (unknown unknowns). They know to measure accuracy, but can't easily ask "why did accuracy drop for zone 42 on Tuesdays?"

## Segmentation: The Critical Practice

Global metrics hide failures.

A model that averages 90% accuracy globally might be 95% accurate for well-served segments (Manhattan, peak hours) and 70% accurate for underserved segments (distant zones, off-peak hours). Without segmentation, the 70% failure is invisible.

Segmentation is the primary tool for detecting disparate impact and is equally critical for operational monitoring: measuring separately by geography, time of day, customer segment, and transaction size reveals hidden problems.

Practice: define meaningful segments upfront. Measure metrics separately for each. Alert if any segment deviates from baseline.

## Retraining Triggers

When should you retrain? Three strategies:

**Scheduled retraining**: Retrain weekly, or monthly, or on a fixed schedule, regardless of performance. Simple to implement. Drawback: you might retrain when nothing has changed, or miss drift because you retrain too infrequently.

**Performance-triggered retraining**: Monitor accuracy on recent data (with ground truth lag). If accuracy drops below threshold, trigger retraining. More responsive to real problems. Drawback: ground truth lag (24-48 hours) means you detect problems late.

**Drift-triggered retraining**: Monitor feature distributions and statistical properties. If drift is detected (via KS test, PSI, or other), trigger retraining before waiting for accuracy to degrade. Fastest response, but requires defining thresholds (what level of drift justifies retraining cost?).

Most production systems use a combination: scheduled retraining on a baseline cadence (weekly), performance monitoring to catch problems earlier, drift detection to be proactive.

## Model Validation Before Deployment

Retraining without validation risks deploying a worse model. Validation must answer: is the new model better than the old one?

**Offline validation**: Train new model on historical data. Compare accuracy on a held-out test set to the current production model's accuracy on the same data. Proceed only if new model is better.

**Relative performance**: Compute accuracy segmented by geography, time, customer type. If new model is better overall but worse for specific segments, decide whether to proceed (maybe the segment is small, or the new model's improvement elsewhere outweighs the loss).

**Shadow mode**: Deploy new model alongside production model. Both see the same requests, but only production model's predictions are served. Compare their predictions; measure new model's accuracy using delayed ground truth. Only promote to production if shadow model looks good.

**Canary deployment**: Route small percentage (1-5%) of traffic to new model. Monitor metrics. If metrics stay healthy, gradually increase percentage. Rollback if metrics degrade.

Netflix's recommendation system uses shadow mode at scale: new models run for weeks in shadow, collecting millions of predictions, before being promoted to production. Cost and time intensive, but catches subtle failures before they impact users.

## Model Versioning and Rollback

Every deployed model must be versioned and kept in the registry. Version identifies:
- Training date
- Dataset used (which rows, which features)
- Hyperparameters
- Training loss, validation accuracy

When a new model is deployed, the old model is retained. If the new model fails in production, rollback is a matter of switching back to the previous version.

Without versioning, rollback means retraining the old model from scratch (slow) or losing the ability to rollback entirely.

Practice: store models in a model registry (MLflow, W&B, custom database). Each model stores metadata: version, training date, metrics, lineage (what data was used). Production always references a specific version ID, not "latest."

## Graceful Degradation in Retraining

Retraining introduces risk: new model might be worse, or retraining might fail. System must continue to serve.

Degradation strategies:
- **Retraining takes too long**: Timeout and stick with current model. Retry later.
- **New model fails validation**: Keep current model, alert engineers. Do not deploy.
- **New model deploys but performs worse**: Canary deployment catches this early. Rollback before full traffic is routed.
- **Retraining pipeline breaks**: Use last-known-good model. Log error. Alert.

The principle: model staleness is bad, but worse models are worse. Err on the side of keeping the current model if the new one is uncertain.

## Feature Stores and Retraining Infrastructure

Retraining requires the same features at training time (offline) and serving time (online). Feature stores (Tecton, Feast, Chronon) centralize feature definitions so offline and online pipelines use identical logic.

Without a feature store, teams manually implement features in two places. Inevitably they diverge: training uses one formula, serving uses another. Model accuracy in training looks good; production accuracy is worse.

Airbnb's Chronon platform reduced time to create a new model from months to weeks by automating the offline/online feature consistency problem.

For your system: ideally, features are computed once and reused. Retraining uses the same feature definitions as serving. This is complex infrastructure; simpler approaches compute features during retraining separately, accepting some risk of drift.

## Operational Metrics for Retraining Success

Measure:
- **Retraining frequency**: How often do you retrain? Is it often enough to catch drift?
- **Time to detect problems**: How long from when data drifts to when retraining is triggered?
- **Time to deploy**: How long from triggering retraining to new model live?
- **Model promotion rate**: What percentage of retrained models are promoted to production vs rejected?
- **Rollback frequency**: How often do you rollback after a bad deployment? (Ideally rare; indicates validation is working)
- **Accuracy by segment**: Is new model better across all segments or just globally?

These metrics indicate whether your retraining system is working or just running.

## Case Study: Data Drift and Late Response

A real scenario: demand prediction model is trained on 2025 data. January 2026 brings unexpected weather. Demand patterns shift. The model was not trained on this pattern.

Timeline:
- Jan 1: Model deployed, working well
- Jan 16: Weather changes, demand patterns shift
- Jan 22: Monitoring detects 5% accuracy drop (if monitoring is implemented)
- Jan 23: Drift tests trigger (if drift detection is implemented)
- Jan 24: Retraining pipeline starts
- Jan 25: New model trained, validated, deployed
- Jan 26: New model reduces error to 2%

Without monitoring, degradation goes undetected until complaints arrive (Jan 28-29). By then, a week of bad predictions have been served.

Prevention: implement drift detection. Retrain within hours of detecting significant shift.

## References


[Understanding Data Drift and Model Drift](https://www.evidentlyai.com/ml-in-production/data-drift)
- Practical guide to detecting covariate and concept drift

[Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) (Sculley et al., Google, NeurIPS 2015)
- Foundational work on retraining, model staleness, and feedback loops

[Netflix Recommendation System: A Data Mining Approach](https://www.slideshare.net/justinpinkul/recsys-2014-netflix-talk)
- Case study: shadow mode for validating new models at scale

[Tecton: ML Feature Platform](https://www.tecton.ai/)
- Solving offline/online feature consistency through centralized feature management

[Airbnb's Chronon: ML Feature Platform for Standardized Feature Engineering](https://www.infoq.com/news/2024/04/airbnb-chronon-open-sourced/)
- How centralized feature definitions solve model retraining challenges

