"""Оценка стоимости автомобилей по сохранённому пайплайну из ноутбука."""
import pickle
from datetime import date
from io import BytesIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import TransformedTargetRegressor

from car_features import (
    EXTRA_NUMERIC, NUMERIC, RAW_COLUMNS, REFERENCE_YEAR, TARGET, model_coefficients,
)

ROOT = Path(__file__).resolve().parent
TEAL = "#176759"
RUST = "#ae5336"
LABELS = {
    "Diesel": "Дизель", "Petrol": "Бензин", "CNG": "Метан (CNG)", "LPG": "Пропан-бутан (LPG)",
    "Individual": "Частное лицо", "Dealer": "Дилер", "Trustmark Dealer": "Сертифицированный дилер",
    "Manual": "Механическая", "Automatic": "Автоматическая", "First Owner": "Первый",
    "Second Owner": "Второй", "Third Owner": "Третий", "Fourth & Above Owner": "Четвёртый и далее",
    "Test Drive Car": "Тест-драйв",
}
NUM_LABELS = {
    "year": "Год выпуска", "km_driven": "Пробег", "mileage": "Топливная эффективность",
    "engine": "Объём двигателя", "max_power": "Мощность", "torque": "Крутящий момент",
    "max_torque_rpm": "Обороты пикового момента", "age_squared": "Квадрат возраста",
    "log_km": "Логарифм пробега", "power_per_litre": "Мощность на литр",
    "age_x_power": "Возраст × мощность", "power_squared": "Квадрат мощности",
    "missing_specs": "Число пропусков",
}
CAT_LABELS = {
    "brand": "Марка", "fuel": "Топливо", "seller_type": "Продавец",
    "transmission": "Коробка", "owner": "Владелец", "seats": "Места",
    "model_family": "Модель", "variant": "Комплектация",
}


@st.cache_resource
def load_model():
    with (ROOT / "car_price.pickle").open("rb") as file:
        return pickle.load(file)


@st.cache_data
def train_data():
    return pd.read_csv(ROOT / "data/cars_train.csv").drop_duplicates(RAW_COLUMNS)


def format_inr(value):
    return f"{value:,.0f}".replace(",", " ") + " ₹"


def read_csv(content):
    """Принять стандартный CSV и файл с разделителем Excel."""
    header = content.splitlines()[0].decode("utf-8-sig") if content.strip() else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(BytesIO(content), sep=sep, encoding="utf-8-sig")


def predict_frame(records):
    missing = [c for c in RAW_COLUMNS if c not in records]
    if missing:
        raise ValueError("Добавьте столбцы " + ", ".join(missing) + ".")
    if records.empty:
        raise ValueError("В файле есть заголовок, но нет строк с автомобилями.")
    frame = records[RAW_COLUMNS].copy()
    for col, low, high, integer in [
        ("year", 1886, date.today().year + 1, True),
        ("km_driven", 0, np.inf, False), ("seats", 1, np.inf, True),
    ]:
        values = pd.to_numeric(frame[col], errors="coerce")
        valid = values.between(low, high) & np.isfinite(values)
        if integer:
            valid &= values.mod(1).eq(0)
        if col == "seats":
            valid |= frame[col].isna()
        if not valid.all():
            positions = (np.flatnonzero(~valid)[:5] + 1).tolist()
            raise ValueError(f"Проверьте {col} в строках данных {positions} (нумерация с 1).")
    if frame["name"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Укажите name для каждого автомобиля.")
    model = load_model()
    with np.errstate(over="ignore"):
        predicted = model.predict(frame)
    if not np.isfinite(predicted).all():
        raise ValueError("Для этих параметров прогноз вышел за числовой диапазон. Проверьте единицы.")
    return np.asarray(predicted)


def explain_year(years):
    out = ~pd.to_numeric(years).between(1983, REFERENCE_YEAR)
    if out.any():
        st.warning(f"У {out.sum()} авто год выпуска за пределами обучающей выборки (1983–2020). "
                   "Модель экстраполирует цену по историческим данным.")


def manual_tab():
    with st.form("single", border=True):
        name = st.text_input("Марка, модель и комплектация", "Maruti Swift Dzire VDI",
                             help="Названия на английском, как в исходных объявлениях.")
        left, middle, right = st.columns(3)
        year = left.number_input("Год выпуска", 1886, date.today().year + 1, 2014)
        km = middle.number_input("Пробег, км", 0, value=145500, step=1000)
        seats = right.number_input("Число мест", 1, 30, 5)
        left, right = st.columns(2)
        fuel = left.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG"], format_func=LABELS.get)
        gear = right.selectbox("Коробка передач", ["Manual", "Automatic"], format_func=LABELS.get)
        seller = left.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"], format_func=LABELS.get)
        owner = right.selectbox("Владелец по счёту", ["First Owner", "Second Owner", "Third Owner",
                                "Fourth & Above Owner", "Test Drive Car"], format_func=LABELS.get)
        with st.expander("Двигатель и расход топлива"):
            st.caption("Поля можно очистить. Для пропущенных характеристик модель использует медианы train.")
            left, middle, right = st.columns(3)
            engine = left.number_input("Объём, см³", 1, value=1248, step=10)
            power = middle.number_input("Мощность, bhp", 0.1, value=74.0)
            mileage = right.number_input("Пробег на единицу топлива", 0.1, value=23.4,
                                         help="Км/л для бензина и дизеля, км/кг для газа.")
            torque = left.number_input("Крутящий момент, Н·м", 0.1, value=190.0)
            rpm = middle.number_input("Верхние обороты момента, об/мин", 1, 10000, 2000)
        submitted = st.form_submit_button("Рассчитать цену", type="primary")
    if submitted:
        torque_text = None if torque is None else f"{torque} Nm"
        if torque_text is not None and rpm is not None:
            torque_text += f" @ {rpm} rpm"
        entry = pd.DataFrame([dict(name=name, year=year, km_driven=km, fuel=fuel, transmission=gear,
                                   seller_type=seller, owner=owner, seats=seats, engine=engine,
                                   max_power=power, mileage=mileage, torque=torque_text)])
        try:
            estimate = predict_frame(entry)[0]
        except ValueError as error:
            st.session_state.pop("estimate", None)
            st.error(str(error))
        else:
            st.session_state.estimate = (name.strip(), year, km, estimate)
    if "estimate" in st.session_state:
        name, year, km, estimate = st.session_state.estimate
        st.metric("Расчётная цена", format_inr(estimate))
        st.caption(f"{name} · {year} г. · {km:,} км".replace(",", " "))
        explain_year(pd.Series([year]))


def batch_tab():
    st.write("Загрузите CSV в UTF-8 с заголовком. Подходят запятая и точка с запятой. "
             "Порядок столбцов произвольный. Цена продажи для расчёта не требуется.")
    example = train_data()[RAW_COLUMNS].head(3).to_csv(index=False)
    st.download_button("Скачать пример CSV", example.encode("utf-8-sig"), "example_cars.csv", "text/csv")
    upload = st.file_uploader("CSV с характеристиками", type="csv")
    if upload is None:
        return
    try:
        records = read_csv(upload.getvalue())
        records["predicted_price_inr"] = predict_frame(records)
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"Не удалось обработать CSV. {error}")
        return
    explain_year(records["year"])
    st.caption(f"Обработано автомобилей — {len(records):,}".replace(",", " "))
    st.dataframe(records[["name", "year", "predicted_price_inr"]], hide_index=True, width="stretch",
                 column_config={"name": "Автомобиль", "year": "Год",
                                "predicted_price_inr": st.column_config.NumberColumn("Оценка, ₹", format="%.0f")})
    st.download_button("Скачать прогнозы", records.to_csv(index=False).encode("utf-8-sig"),
                       "car_predictions.csv", "text/csv", type="primary")


def market_tab():
    data = train_data()
    st.subheader("Рынок в обучающей выборке")
    st.caption("Объявления после удаления повторов по признакам. Цены в индийских рупиях.")
    a, b, c = st.columns(3)
    a.metric("Автомобили", f"{len(data):,}".replace(",", " "))
    b.metric("Медианная цена", format_inr(data[TARGET].median()))
    c.metric("Годы выпуска", f"{data.year.min()}–{data.year.max()}")

    st.markdown("#### Распределение цен")
    edges = np.geomspace(data[TARGET].min(), data[TARGET].max(), 31)
    counts, _ = np.histogram(data[TARGET], bins=edges)
    bins = pd.DataFrame({"от": edges[:-1], "до": edges[1:], "объявления": counts})
    chart = alt.Chart(bins).mark_bar(color=TEAL).encode(
        x=alt.X("от:Q", title="Цена, ₹ · логарифмическая шкала",
                scale=alt.Scale(type="log", domain=[edges[0], edges[-1]]), axis=alt.Axis(format="~s")),
        x2="до:Q", y=alt.Y("объявления:Q", title="Автомобили"),
        tooltip=[alt.Tooltip("от:Q", format=",.0f", title="От, ₹"),
                 alt.Tooltip("до:Q", format=",.0f", title="До, ₹"), "объявления:Q"],
    ).properties(height=270)
    st.altair_chart(chart, width="stretch")

    st.markdown("#### Цена по году выпуска")
    yearly = data.groupby("year")[TARGET].agg(
        median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), count="size"
    ).reset_index()
    base = alt.Chart(yearly).encode(x=alt.X("year:Q", title="Год выпуска",
                                            scale=alt.Scale(zero=False), axis=alt.Axis(format="d")))
    band = base.mark_area(color=TEAL, opacity=.15).encode(y=alt.Y("q25:Q", title="Цена, ₹",
            axis=alt.Axis(format="~s")), y2="q75:Q")
    line = base.mark_line(color=TEAL, point=True).encode(y="median:Q", tooltip=[
        alt.Tooltip("year:Q", title="Год", format="d"), alt.Tooltip("count:Q", title="Объявления"),
        alt.Tooltip("median:Q", title="Медиана, ₹", format=",.0f"),
        alt.Tooltip("q25:Q", title="25%, ₹", format=",.0f"),
        alt.Tooltip("q75:Q", title="75%, ₹", format=",.0f")])
    st.altair_chart((band + line).properties(height=300), width="stretch")
    st.caption("Линия показывает медиану, полоса охватывает центральные 50% цен каждого года.")


def feature_label(name):
    if name in NUM_LABELS:
        return NUM_LABELS[name]
    for prefix, label in CAT_LABELS.items():
        if name.startswith(prefix + "_"):
            value = name[len(prefix) + 1:].replace("infrequent_sklearn", "редкие")
            return f"{label} · {value}"
    return name


def weights_tab():
    model = load_model()
    coef = model_coefficients(model)
    log_target = isinstance(model.named_steps["regression"], TransformedTargetRegressor)
    st.subheader("Коэффициенты Ridge")
    if log_target:
        st.write("Модель оценивает логарифм цены. У числовых признаков вес соответствует изменению на одно "
                 "стандартное отклонение. Вес 0,10 соответствует множителю exp(0,10) ≈ 1,105 при прочих равных.")
    else:
        st.write("Веса числовых признаков выражают изменение оценки в рупиях при увеличении признака "
                 "на одно стандартное отклонение.")
    st.caption("Для сравнения двух уровней категории берите разницу их коэффициентов. Связанные признаки "
               "могут делить между собой влияние на прогноз.")
    df = coef.rename_axis("feature").reset_index()
    df["Признак"] = df["feature"].map(feature_label)
    df["Тип"] = np.where(df.feature.isin(NUMERIC + EXTRA_NUMERIC), "Числовые", "Категории")
    group = st.radio("Группа признаков", ["Все", "Числовые", "Категории"], horizontal=True)
    filtered = df if group == "Все" else df[df["Тип"] == group]
    ranked = filtered.sort_values("coefficient", key=abs, ascending=False)
    top = ranked.head(15)
    chart = alt.Chart(top).mark_bar().encode(
        y=alt.Y("Признак:N", sort=alt.SortField("coefficient", order="descending"), title=None,
                axis=alt.Axis(labelLimit=280)),
        x=alt.X("coefficient:Q", title="Коэффициент в логарифме цены" if log_target else "Коэффициент, ₹"),
        color=alt.condition("datum.coefficient >= 0", alt.value(TEAL), alt.value(RUST)),
        tooltip=["Признак:N", "feature:N", alt.Tooltip("coefficient:Q", title="Вес", format=".4f")],
    ).properties(height=max(220, len(top)*25))
    st.altair_chart(chart, width="stretch")
    st.caption(f"Показано {len(top)} из {len(ranked)} признаков с наибольшим весом по модулю.")
    with st.expander("Все коэффициенты"):
        st.dataframe(ranked[["Признак", "feature", "coefficient"]], hide_index=True, width="stretch",
                     column_config={"feature": "Имя в модели",
                                    "coefficient": st.column_config.NumberColumn("Вес", format="%.5f")})


def main():
    st.set_page_config(page_title="Оценка автомобиля · CarDekho", page_icon="🚘", layout="centered")
    st.title("Оценка автомобиля")
    st.caption("Рынок подержанных машин в Индии · модель по объявлениям 1983–2020 годов выпуска")
    estimate, market, weights = st.tabs(["Прогноз", "Данные рынка", "Веса модели"])
    with estimate:
        mode = st.radio("Способ ввода", ["Один автомобиль", "Файл CSV"], horizontal=True, label_visibility="collapsed")
        if mode == "Один автомобиль":
            manual_tab()
        else:
            batch_tab()
    with market:
        market_tab()
    with weights:
        weights_tab()


if __name__ == "__main__":
    main()
