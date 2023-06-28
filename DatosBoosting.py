import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/Dunido.csv')
    x = dt_heart.drop(['INCIDENCIA'], axis=1)
    y = dt_heart['INCIDENCIA']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)

    # Resultados con los datos originales
    boosting = GradientBoostingClassifier(loss='exponential', learning_rate=0.15, n_estimators=188, max_depth=5).fit(X_train, y_train)
    boosting_pred = boosting.predict(X_test)
    print('=' * 64)
    print('Datos Originales')
    print(accuracy_score(boosting_pred, y_test))

    # Resultados con los datos normalizados
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    boosting_normalized = GradientBoostingClassifier(n_estimators=188).fit(X_train_normalized, y_train)
    boosting_pred_normalized = boosting_normalized.predict(X_test_normalized)
    print('=' * 64)
    print('Datos Normalizados')
    print(accuracy_score(boosting_pred_normalized, y_test))

    # Resultados con los datos normalizados y discretizados
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_discretized = discretizer.fit_transform(X_train_normalized)
    X_test_discretized = discretizer.transform(X_test_normalized)

    boosting_discretized = GradientBoostingClassifier(n_estimators=188).fit(X_train_discretized, y_train)
    boosting_pred_discretized = boosting_discretized.predict(X_test_discretized)
    print('=' * 64)
    print('Datos Normalizados y Discretizados')
    print(accuracy_score(boosting_pred_discretized, y_test))
