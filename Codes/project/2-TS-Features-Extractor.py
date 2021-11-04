print("[INFO] importing libraries ...")
import os
import numpy as np
import pandas as pd
import config as cfg
import pickle
import tsfel
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import seasonal, stattools

print("[INFO] importing pickles files....")
with open(os.path.join(cfg.pickle_dir, "df_sum.pickle"), "rb") as handle:
    df_sum = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_max.pickle"), "rb") as handle:
    df_max = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_xCe.pickle"), "rb") as handle:
    df_xCe = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_yCe.pickle"), "rb") as handle:
    df_yCe = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_label.pickle"), "rb") as handle:
    df_label = pickle.load(handle)


###########################################################################################################
###########################################################################################################
###########################################################################################################
# orders = [2, 0, 0]
# index = ["Sample_" + str(i) for i in range(1744)]

# FX = np.zeros((1744, 4 * orders[0]))

# for i in range(1744):  # 1744
#     model = ARIMA(df_sum[index[i]].dropna(), order=orders)
#     model_fitted = model.fit(disp=-1)
#     FX[i, 0 : orders[0]] = model_fitted.arparams

#     model = ARIMA(df_max[index[i]].dropna(), order=orders)
#     model_fitted = model.fit(disp=-1)
#     FX[i, orders[0] : 2 * orders[0]] = model_fitted.arparams

#     model = ARIMA(df_xCe[index[i]].dropna(), order=orders)
#     model_fitted = model.fit(disp=-1)
#     FX[i, 2 * orders[0] : 3 * orders[0]] = model_fitted.arparams

#     model = ARIMA(df_yCe[index[i]].dropna(), order=orders)
#     model_fitted = model.fit(disp=-1)
#     FX[i, 3 * orders[0] : 4 * orders[0]] = model_fitted.arparams

# col = [
#     "AR1sum",
#     "AR2sum",
#     "AR1max",
#     "AR2max",
#     "AR1xCe",
#     "AR2xCe",
#     "AR1yCe",
#     "AR2yCe",
# ]
# df_AR_features = pd.DataFrame(
#     FX,
#     index=index,
#     columns=col,
# )
# with open(os.path.join(cfg.pickle_dir, "df_AR_features.txt"), "w") as handle:
#     handle.write(df_AR_features.to_string())
# with open(os.path.join(cfg.pickle_dir, "df_AR_features.pickle"), "wb") as handle:
#     pickle.dump(df_AR_features, handle)

###########################################################################################################
###########################################################################################################
###########################################################################################################

X = np.zeros((1744, 18))
Y = np.zeros((1744, 36))
Z = np.zeros((1744, 131))
index = ["Sample_" + str(i) for i in range(1744)]


print("[INFO] Extracting features...")
for i in range(1744):
    # windows = tsfel.signal_window_splitter(df_sum[index[i]], 30, 0.1)

    X[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("temporal"), df_sum[index[i]]
    ).values
    Y[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("statistical"), df_sum[index[i]]
    ).values
    Z[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("spectral"), df_sum[index[i]]
    ).values

col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("temporal"), df_sum["Sample_0"]
).columns
df_sum_temporal_features = pd.DataFrame(X, index=index, columns=col)
col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("statistical"), df_sum["Sample_0"]
).columns
df_sum_statistical_features = pd.DataFrame(Y, index=index, columns=col)
col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("spectral"), df_sum["Sample_0"]
).columns
df_sum_spectral_features = pd.DataFrame(Z, index=index, columns=col)

with open(os.path.join(cfg.pickle_dir, "df_sum_temporal_features1.txt"), "w") as handle:
    handle.write(df_sum_temporal_features.to_string())
with open(
    os.path.join(cfg.pickle_dir, "df_sum_statistical_features1.txt"), "w"
) as handle:
    handle.write(df_sum_statistical_features.to_string())
with open(os.path.join(cfg.pickle_dir, "df_sum_spectral_features1.txt"), "w") as handle:
    handle.write(df_sum_spectral_features.to_string())

with open(
    os.path.join(cfg.pickle_dir, "df_sum_temporal_features.pickle"), "wb"
) as handle:
    pickle.dump(df_sum_temporal_features, handle)
with open(
    os.path.join(cfg.pickle_dir, "df_sum_statistical_features.pickle"), "wb"
) as handle:
    pickle.dump(df_sum_statistical_features, handle)
with open(
    os.path.join(cfg.pickle_dir, "df_sum_spectral_features.pickle"), "wb"
) as handle:
    pickle.dump(df_sum_spectral_features, handle)


###########################################################################################################
###########################################################################################################
###########################################################################################################

for i in range(1744):
    X[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("temporal"), df_max[index[i]]
    ).values
    Y[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("statistical"), df_max[index[i]]
    ).values
    Z[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("spectral"), df_max[index[i]]
    ).values

col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("temporal"), df_max["Sample_0"]
).columns
df_max_temporal_features = pd.DataFrame(X, index=index, columns=col)
col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("statistical"), df_max["Sample_0"]
).columns
df_max_statistical_features = pd.DataFrame(Y, index=index, columns=col)
col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("spectral"), df_max["Sample_0"]
).columns
df_max_spectral_features = pd.DataFrame(Z, index=index, columns=col)

with open(os.path.join(cfg.pickle_dir, "df_max_temporal_features.txt"), "w") as handle:
    handle.write(df_max_temporal_features.to_string())
with open(
    os.path.join(cfg.pickle_dir, "df_max_statistical_features.txt"), "w"
) as handle:
    handle.write(df_max_statistical_features.to_string())
with open(os.path.join(cfg.pickle_dir, "df_max_spectral_features.txt"), "w") as handle:
    handle.write(df_max_spectral_features.to_string())

with open(
    os.path.join(cfg.pickle_dir, "df_max_temporal_features.pickle"), "wb"
) as handle:
    pickle.dump(df_max_temporal_features, handle)
with open(
    os.path.join(cfg.pickle_dir, "df_max_statistical_features.pickle"), "wb"
) as handle:
    pickle.dump(df_max_statistical_features, handle)
with open(
    os.path.join(cfg.pickle_dir, "df_max_spectral_features.pickle"), "wb"
) as handle:
    pickle.dump(df_max_spectral_features, handle)


###########################################################################################################
###########################################################################################################
###########################################################################################################

for i in range(1744):
    X[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("temporal"), df_xCe[index[i]]
    ).values
    Y[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("statistical"), df_xCe[index[i]]
    ).values
    Z[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("spectral"), df_xCe[index[i]]
    ).values

col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("temporal"), df_xCe["Sample_0"]
).columns
df_xCe_temporal_features = pd.DataFrame(X, index=index, columns=col)
col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("statistical"), df_xCe["Sample_0"]
).columns
df_xCe_statistical_features = pd.DataFrame(Y, index=index, columns=col)
col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("spectral"), df_xCe["Sample_0"]
).columns
df_xCe_spectral_features = pd.DataFrame(Z, index=index, columns=col)

with open(os.path.join(cfg.pickle_dir, "df_xCe_temporal_features.txt"), "w") as handle:
    handle.write(df_xCe_temporal_features.to_string())
with open(
    os.path.join(cfg.pickle_dir, "df_xCe_statistical_features.txt"), "w"
) as handle:
    handle.write(df_xCe_statistical_features.to_string())
with open(os.path.join(cfg.pickle_dir, "df_xCe_spectral_features.txt"), "w") as handle:
    handle.write(df_xCe_spectral_features.to_string())

with open(
    os.path.join(cfg.pickle_dir, "df_xCe_temporal_features.pickle"), "wb"
) as handle:
    pickle.dump(df_xCe_temporal_features, handle)
with open(
    os.path.join(cfg.pickle_dir, "df_xCe_statistical_features.pickle"), "wb"
) as handle:
    pickle.dump(df_xCe_statistical_features, handle)
with open(
    os.path.join(cfg.pickle_dir, "df_xCe_spectral_features.pickle"), "wb"
) as handle:
    pickle.dump(df_xCe_spectral_features, handle)


###########################################################################################################
###########################################################################################################
###########################################################################################################

for i in range(1744):
    X[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("temporal"), df_yCe[index[i]]
    ).values
    Y[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("statistical"), df_yCe[index[i]]
    ).values
    Z[i, :] = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain("spectral"), df_yCe[index[i]]
    ).values

col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("temporal"), df_yCe["Sample_0"]
).columns
df_yCe_temporal_features = pd.DataFrame(X, index=index, columns=col)
col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("statistical"), df_yCe["Sample_0"]
).columns
df_yCe_statistical_features = pd.DataFrame(Y, index=index, columns=col)
col = tsfel.time_series_features_extractor(
    tsfel.get_features_by_domain("spectral"), df_yCe["Sample_0"]
).columns
df_yCe_spectral_features = pd.DataFrame(Z, index=index, columns=col)

with open(os.path.join(cfg.pickle_dir, "df_yCe_temporal_features.txt"), "w") as handle:
    handle.write(df_yCe_temporal_features.to_string())
with open(
    os.path.join(cfg.pickle_dir, "df_yCe_statistical_features.txt"), "w"
) as handle:
    handle.write(df_yCe_statistical_features.to_string())
with open(os.path.join(cfg.pickle_dir, "df_yCe_spectral_features.txt"), "w") as handle:
    handle.write(df_yCe_spectral_features.to_string())

with open(
    os.path.join(cfg.pickle_dir, "df_yCe_temporal_features.pickle"), "wb"
) as handle:
    pickle.dump(df_yCe_temporal_features, handle)
with open(
    os.path.join(cfg.pickle_dir, "df_yCe_statistical_features.pickle"), "wb"
) as handle:
    pickle.dump(df_yCe_statistical_features, handle)
with open(
    os.path.join(cfg.pickle_dir, "df_yCe_spectral_features.pickle"), "wb"
) as handle:
    pickle.dump(df_yCe_spectral_features, handle)
