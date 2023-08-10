from src.lib.plotting import (
    get_figsize,
    load_hist,
    get_confidence_interval,
    save_plot,
    get_average,
)
import matplotlib.pyplot as plt
import pandas as pd
import pprint

hist_bin = load_hist("LOB-BIN-experiment-final")
hist_dain = load_hist("LOB-DAIN-experiment-final")
hist_edain_local = load_hist("LOB-EDAIN-experiment-final-v1")
hist_edain_global = load_hist("LOB-EDAIN-global-experiment-final-v1")
hist_edain_kl = load_hist("LOB-EDAIN-KL-experiment-final-v1")
hist_min_max = load_hist("LOB-min-max-experiment-final")
hist_standard_scaling = load_hist("LOB-standard-scaling-experiment-final")

histories = [hist_standard_scaling, hist_min_max, hist_bin, hist_dain, hist_edain_local, hist_edain_global, hist_edain_kl]
names = ['Standard scaling', 'Min-max scaling', 'BIN', 'DAIN', 'EDAIN (local-aware)', 'EDAIN (global-aware)', 'EDAIN-KL']

df = pd.DataFrame(columns=['Method', 'Cohen\'s Kappa', 'Average $F_1$-score'])
for i, (hist, lab) in enumerate(zip(histories, names)):
    m, s = get_confidence_interval(hist, key='kappa')
    m2, s2 = get_confidence_interval(hist, key='f1_avg')
    df.loc[i] = [lab, f"${m:.4f} \pm {s:.4f}$", f"${m2:.4f} \pm {s2:.4f}$"]
print(df.to_latex(index=False))
