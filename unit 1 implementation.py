import pandas as pd, numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score

df = pd.read_csv("/mnt/user-data/outputs/urban_noise_pollution_dataset.csv")

print("=== Descriptive stats for noise_level_db ===")
print(df["noise_level_db"].describe())

print("\n=== Correlation with target (numeric features) ===")
num_cols = df.select_dtypes(include=np.number).columns.drop(["record_id","location_id"])
corr = df[num_cols].corr()["noise_level_db"].sort_values(ascending=False)
print(corr)

print("\n=== ANOVA: noise_level_db across road_type ===")
groups = [g["noise_level_db"].values for _, g in df.groupby("road_type")]
f, p = stats.f_oneway(*groups)
print("F =", round(f,2), "p =", p)

print("\n=== Chi-square: road_type vs noise_category ===")
ct = pd.crosstab(df["road_type"], df["noise_category"])
chi2, p2, dof, exp = stats.chi2_contingency(ct)
print("chi2 =", round(chi2,2), "p =", p2)

print("\n=== t-test: weekend vs weekday noise ===")
a = df[df.is_weekend==1]["noise_level_db"]
b = df[df.is_weekend==0]["noise_level_db"]
t, pt = stats.ttest_ind(a, b)
print("t =", round(t,3), "p =", pt, "mean_weekend=", round(a.mean(),2), "mean_weekday=", round(b.mean(),2))

print("\n=== Quick RandomForest model (regression) ===")
feat_cols = ["traffic_volume_vph","heavy_vehicle_percentage","two_wheeler_percentage",
             "construction_intensity","population_density_per_km2","event_attendance_estimate",
             "distance_to_highway_km","green_cover_percentage","historical_avg_noise_db",
             "hour","is_weekend","road_type","land_use_type"]
X = df[feat_cols].copy()
for c in ["road_type","land_use_type"]:
    X[c] = LabelEncoder().fit_transform(X[c])
y = df["noise_level_db"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
m.fit(Xtr, ytr)
pred = m.predict(Xte)
print("R2 =", round(r2_score(yte, pred),3), "MAE =", round(mean_absolute_error(yte, pred),2))
imp = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\nFeature importances:\n", imp)

print("\n=== Quick RandomForest model (classification: noise_category) ===")
yc = df["noise_category"]
Xtr, Xte, ytr, yte = train_test_split(X, yc, test_size=0.2, random_state=42, stratify=yc)
mc = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
mc.fit(Xtr, ytr)
predc = mc.predict(Xte)
print("Accuracy =", round(accuracy_score(yte, predc),3), "F1(weighted) =", round(f1_score(yte, predc, average='weighted'),3))
