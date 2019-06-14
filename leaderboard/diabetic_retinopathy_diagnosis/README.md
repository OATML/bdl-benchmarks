# Diabetic Retinopathy Diagnosis
The baseline results we evaluated on this benchmark are ranked below by AUC@50% data retained:

| Method              | AUC<br>(50% data retained)  | Accuracy<br>(50% data retained) | AUC<br>(100% data retained) | Accuracy<br>(100% data retained) |
| ------------------- | :-------------------------: | :-----------------------------: | :-------------------------: | :-------------------------------: |
| Ensemble MC Dropout | 88.1 ± 1.2                  | 92.4 ± 0.9                      | 82.5 ± 1.1                  | 85.3 ± 1.0                        |
| MC Dropout          | 87.8 ± 1.1                  | 91.3 ± 0.7                      | 82.1 ± 0.9                  | 84.5 ± 0.9                        |
| Deep Ensembles      | 87.2 ± 0.9                  | 89.9 ± 0.9                      | 81.8 ± 1.1                  | 84.6 ± 0.7                        |
| Mean-field VI       | 86.6 ± 1.1                  | 88.1 ± 1.1                      | 82.1 ± 1.2                  | 84.3 ± 0.7                        |
| Deterministic       | 84.9 ± 1.1                  | 86.1 ± 0.6                      | 82.0 ± 1.0                  | 84.2 ± 0.6                        |
| Random              | 81.8 ± 1.2                  | 84.8 ± 0.9                      | 82.0 ± 0.9                  | 84.2 ± 0.5                        |

These are also plotted below in an area under the receiver-operating characteristic curve (AUC) and binary accuracy for the different baselines. The methods that capture uncertainty score better when less data is retained, referring the least certain patients to expert doctors. The best scoring methods, _MC Dropout_, _mean-field variational inference_ and _Deep Ensembles_, estimate and use the predictive uncertainty. The deterministic deep model regularized by _standard dropout_ uses only aleatoric uncertainty and performs worse. Shading shows the standard error.

<p align="center">
<img src="http://www.cs.ox.ac.uk/people/angelos.filos/assets/bdl-benchmarks/metrics/auc.png" style="float: left; width: 30%; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em" >
<img src="http://www.cs.ox.ac.uk/people/angelos.filos/assets/bdl-benchmarks/metrics/accuracy.png" style="float: left; width: 30%; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em" >
<img src="http://www.cs.ox.ac.uk/people/angelos.filos/assets/bdl-benchmarks/metrics/legend.png" style="float: left; width: 100%; margin-bottom: 0.5em; margin-top: 0.0em" >
</p>
