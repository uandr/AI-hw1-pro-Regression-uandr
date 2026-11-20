"""Подготовка записей CarDekho для EDA, обучения и прогноза."""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "selling_price"
RAW_COLUMNS = [
    "name", "year", "km_driven", "fuel", "seller_type", "transmission",
    "owner", "mileage", "engine", "max_power", "torque", "seats",
]
NUMERIC = [
    "year", "km_driven", "mileage", "engine", "max_power",
    "torque", "max_torque_rpm", "seats",
]
CATEGORIES = ["brand", "fuel", "seller_type", "transmission", "owner"]
EXTRA_NUMERIC = [
    "age_squared", "log_km", "power_per_litre", "age_x_power",
    "power_squared", "missing_specs",
]
REFERENCE_YEAR = 2020  # Последний модельный год в обучающей выборке
NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
TORQUE_SPLIT = re.compile(r"@|\bat\b|/", re.I)
KG_M = re.compile(r"(?:kgf?\s*[.·-]?\s*m|mkg)", re.I)


@lru_cache(maxsize=4096)
def parse_torque(value):
    """Вернуть момент в Н·м и верхние обороты полки в об/мин."""
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip().lower().replace(",", "").replace("±", "+/-")
    parts = TORQUE_SPLIT.split(text, maxsplit=1)
    match = re.match(NUMBER, parts[0])
    if not match:
        return np.nan, np.nan
    torque = float(match.group())
    rpm = np.nan
    if len(parts) == 2:
        rpm_text = parts[1].split("(")[0]
        rpm_values = [float(n) for n in re.findall(NUMBER, rpm_text)]
        if rpm_values:
            rpm = sum(rpm_values[:2]) if "+/-" in rpm_text else max(rpm_values)

    # В формате «380Nm(38.7kgm)» первое число уже дано в Н·м
    first_unit = parts[0].split("(")[0]
    if KG_M.search(first_unit):
        torque *= 9.80665
    elif KG_M.search(text) and "nm" not in first_unit:
        # «115@ 2500(kgm@ rpm)» даёт неправдоподобные 1128 Н·м
        # При конфликте обозначения и порядка величины единица неоднозначна
        torque = torque * 9.80665 if torque < 100 else np.nan
    if torque <= 0:
        torque = np.nan
    if not 0 < rpm <= 10000:
        rpm = np.nan
    return torque, rpm


def clean_measurements(raw):
    """Разобрать единицы без обучения. Цена, если она есть, сохраняется"""
    frame = raw.copy()
    for col in ["mileage", "engine", "max_power"]:
        numbers = frame[col].astype("string").str.extract(
            rf"^\s*\+?({NUMBER})(?=\s|[a-zA-Z]|$)", expand=False
        )
        values = pd.to_numeric(numbers, errors="coerce").astype(float)
        frame[col] = values.where(values > 0)
    for col in ["year", "km_driven", "seats"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    frame[["torque", "max_torque_rpm"]] = pd.DataFrame(
        [parse_torque(v) for v in frame["torque"]], index=frame.index,
        columns=["torque", "max_torque_rpm"], dtype=float,
    )
    return frame


def feature_frame(raw, stage="engineered"):
    """Построить только признаки одного объекта, без статистик по выборке"""
    frame = clean_measurements(raw[RAW_COLUMNS])
    name = frame.pop("name").astype("string").str.lower().str.split().str.join(" ")
    words = name.str.split()
    frame["brand"] = words.str[0].replace({"land": "land rover"})
    if stage in {"names", "engineered"}:
        frame["model_family"] = words.str[:2].str.join(" ")
        frame["variant"] = name
    if stage == "engineered":
        age = REFERENCE_YEAR - frame["year"]
        frame["age_squared"] = age ** 2
        frame["log_km"] = np.log1p(frame["km_driven"])
        frame["power_per_litre"] = 1000 * frame["max_power"] / frame["engine"]
        frame["age_x_power"] = age * frame["max_power"]
        frame["power_squared"] = frame["max_power"] ** 2
        frame["missing_specs"] = frame[
            ["mileage", "engine", "max_power", "torque", "max_torque_rpm", "seats"]
        ].isna().sum(axis=1)
    cats = CATEGORIES + (["model_family", "variant"] if stage != "brand" else [])
    for col in cats:
        frame[col] = frame[col].astype("string").str.strip().str.lower().fillna("unknown").astype(object)
    return frame


def make_preprocessor(stage="engineered"):
    """Медианы, масштаб и категории обучаются отдельно в каждом train-фолде."""
    numeric = [col for col in NUMERIC if col != "seats"]
    if stage == "engineered":
        numeric += EXTRA_NUMERIC
    numeric_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    # В базовом эксперименте задание требует разобрать эффект опорного уровня.
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore")
    transforms = [
        ("num", numeric_steps, numeric),
        ("cat", ohe, CATEGORIES),
        ("seats", make_pipeline(SimpleImputer(strategy="median"), ohe), ["seats"]),
    ]
    if stage != "brand":
        # Редкие комплектации объединяем. Новая попадёт в эту же группу.
        transforms.append(("names", OneHotEncoder(
            min_frequency=3, handle_unknown="infrequent_if_exist",
        ), ["model_family", "variant"]))
    return ColumnTransformer(transforms, verbose_feature_names_out=False)


def model_coefficients(model):
    """Извлечь имена в точном порядке матрицы, использованной Ridge."""
    fitted = model.named_steps["regression"]
    if hasattr(fitted, "regressor_"):  # TransformedTargetRegressor
        fitted = fitted.regressor_
    prep, ridge = fitted.named_steps["prep"], fitted.named_steps["ridge"]
    return pd.Series(ridge.coef_, index=prep.get_feature_names_out(), name="coefficient")
