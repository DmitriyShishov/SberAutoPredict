import pandas as pd
import dill
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def filter_data(df):
    df = df.copy()

    columns_to_drop = [
        'session_id',
        'client_id',
        'utm_keyword',
        'device_model'
    ]
    return df.drop(columns_to_drop, axis=1)


def fill_specific_nan(df):
    df = df.copy()
# Объектам с неопределенным значением ОС и бренда присваиваем значение other
    df.loc[(df["device_os"].isna()) & (df["device_brand"].isna()), ['device_os', 'device_brand']] = 'other'

# Для устройств, работающих на ОС Macintosh или iOS, заполняем пропущенные значения в столбце бренда значением Apple
    df.loc[(df["device_brand"].isna()) & ((df["device_os"] == 'Macintosh') |
                                          (df["device_os"] == 'iOS')), 'device_brand'] = 'Apple'

# Остальные пропуски в столбце Бренд заполняем значением other
    df.device_brand = df.device_brand.fillna('other')

# Для мобильных устройств и планшетов бренда Apple заполняем пропуски в столбце ОС значением iOS
    df.loc[(df["device_os"].isna()) & (df["device_brand"] == 'Apple') &
           (df["device_category"] != 'desktop'), 'device_os'] = 'iOS'

# Для стационарных компьютеров бренда Apple заполняем пропуски в столбце ОС значением iOS
    df.loc[(df["device_os"].isna()) & (df["device_brand"] == 'Apple') &
           (df["device_category"] == 'desktop'), 'device_os'] = 'Macintosh'

# Учитывая, что почти 90 процентов всех мобильных устройств и планшетов работают на ОС iOS или Android,
# для всех остальных брендов заполняем пропуски в столбце ОС значением Android
    df.loc[(df["device_os"].isna()) & (df["device_category"] != 'desktop'), 'device_os'] = 'Android'

# Учитывая, что более 90 процентов всех стационарных компьюетов работают на ОС Macintosh или Windows,
# для всех остальных брендов заполняем пропуски в столбце ОС значением Windows
    df.loc[(df["device_os"].isna()) & (df["device_category"] == 'desktop'), 'device_os'] = 'Windows'

# При отсутствии дополнительной информации данного признака и относительной сбалансировнности значений в нем,
    # заполняем пропуски значением other
    df.utm_campaign = df.utm_campaign.fillna('other')

    return df


def remove_outliers(df):
    # Заполняем аномальные значения в признаке номера визита верхними и нижними порогами,
    # используя интерквартильный размах

    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        return boundaries

    df = df.copy()

    boundaries = calculate_outliers(df['visit_number'])

    df.loc[df['visit_number'] < boundaries[0], 'visit_number'] = round(boundaries[0])
    df.loc[df['visit_number'] > boundaries[1], 'visit_number'] = round(boundaries[1])

    return df


def create_features(df):
    df = df.copy()
# Объединяем столбцы даты и времени визита, получаем новые признаки, которые могут повлиять на ЦД.
    df['visit_date_time'] = df['visit_date'] + ' ' + df['visit_time']
    df['visit_date_time'] = pd.to_datetime(df.visit_date_time, utc=True)

# Выявляем месяц, день недели и время суток, в которое клиент совершил действие на сайте.
    df['month'] = df.visit_date_time.apply(lambda x: int(x.month))
    df['dayofweek'] = df.visit_date_time.apply(lambda x: int(x.dayofweek))
    df['daytime'] = df.visit_date_time.apply(lambda x: int(x.hour))

    df.daytime = df.daytime.apply(lambda x: 'night' if x in range(0, 6) else
                                            'morning' if x in range(6, 12) else
                                            'day' if x in range(12, 18) else 'evening')

# Имея информацию о перечне источников привлечения из социальных сетей, создадим новый признак Source_type
    social_net_source_list = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                              'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df['utm_source_type'] = df.apply(
        lambda x: 'social_net' if x['utm_source'] in social_net_source_list else 'other', axis=1)

# Имея информацию о перечне типа привлечения, создадим новый признак Medium type
    organic_medium_list = ['organic', 'referral', '(none)']
    df['utm_medium_type'] = df.apply(lambda x: 'organic' if x['utm_medium'] in organic_medium_list else 'other', axis=1)

    return df.drop(['visit_date', 'visit_time', 'visit_date_time'], axis=1)


def device_screen_func(df):
    df = df.copy()
# Заменим столбец с разрешением экрана столбцом, описывающий площадь экрана. Создадим признаки длины и ширины экрана,
    # обработаем аномальные значения для каждого типа устройств, затем создадим столбец площади экрана
    df['device_screen_resolution'] = (df['device_screen_resolution'].
                                      apply(lambda x: x if x != '(not set)' else '414x896'))

    df['device_screen_a'] = (df['device_screen_resolution'].apply(lambda x: x.lower().split('x')).
                             apply(lambda x: int(x[0])))
    df['device_screen_b'] = (df['device_screen_resolution'].apply(lambda x: x.lower().split('x')).
                             apply(lambda x: int(x[1])))
    df['device_screen_a'] = df.apply(lambda x: 360 if ((x.device_category == 'mobile') and
                                                       (x.device_screen_a == 0)) else x.device_screen_a, axis=1)
    df['device_screen_b'] = df.apply(lambda x: 896 if ((x.device_category == 'mobile') and
                                                       (x.device_screen_b == 0)) else x.device_screen_b, axis=1)
    df['device_screen_a'] = df.apply(lambda x: 1920 if ((x.device_category == 'desktop') and
                                                        (x.device_screen_a == 0)) else x.device_screen_a, axis=1)
    df['device_screen_b'] = df.apply(lambda x: 1080 if ((x.device_category == 'desktop') and
                                                        (x.device_screen_b == 0)) else x.device_screen_b, axis=1)
    df['device_screen_b'] = df.apply(lambda x: 2000 if ((x.device_category == 'desktop') and
                                                        (x.device_screen_b == 20000)) else x.device_screen_b, axis=1)
    df['device_screen_area'] = df.device_screen_a * df.device_screen_b

    return df.drop(['device_screen_a', 'device_screen_b'], axis=1)


def recalculation(df):
    # Для оптимизации мощностей принято решение уменьшить количество уникальных значений категориальных признаков.
    # В данных столбцах заменены значения, встречающиеся менее, чем в 0,05% объектов, на значение other.
    column_list_recalc = ['utm_source', 'utm_campaign', 'utm_adcontent', 'device_brand', 'geo_city', 'geo_country']

    for column in column_list_recalc:
        column_prop = df[column].value_counts(normalize=True).apply(lambda x: 100 * float(f"{x: 0.4f}"))
        column_prop_list = column_prop[column_prop > 0.05].keys().tolist()
        df[column] = df[column].apply(lambda x: x if x in column_prop_list else 'other')
        del column_prop, column_prop_list

    return df


def main():
    print('SberAutoPodpiska Prediction Pipeline')

    df = pd.read_csv('data/df_session_with_target.csv', low_memory=False)

    x = df.drop(['target'], axis=1)
    y = df['target']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor_func = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('specific_fill_nan', FunctionTransformer(fill_specific_nan)),
        ('outliers_remover', FunctionTransformer(remove_outliers)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('device_screen_operations', FunctionTransformer(device_screen_func)),
        ('recalculation', FunctionTransformer(recalculation)),
        ('column_transformer', column_transformer)
    ])

    models = [
        LogisticRegression(solver='liblinear', class_weight='balanced'),
        RandomForestClassifier(class_weight='balanced')
    ]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor_func),
            ('classifier', model)
            ])

        pipe.fit(x_train, y_train)
        roc_auc = roc_auc_score(y_test, pipe.predict_proba(x_test)[:, 1])

        print(f'model: {type(model).__name__}, ROC_AUC: {roc_auc:.4f}')

        if roc_auc > best_score:
            best_score = roc_auc
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, ROC_AUC: {best_score:.4f}')
    best_pipe.fit(x, y)

    with open('model/sber_auto_prod.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Sber Auto prediction model',
                'author': 'Dmitry Shishov',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file, recurse=True)

    print('Model is saved')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
