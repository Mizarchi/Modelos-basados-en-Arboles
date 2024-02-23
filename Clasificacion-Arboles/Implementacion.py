#11

import pandas as pd
from clasificacion11y12 import graficar_distribucion_clases_1, particion_estratificada_1, ajustar_arbol_decision_1

# Cargar el DataFrame con los datos procesados
df = pd.read_csv('datos_procesados.csv')

# Utilizar las funciones de clasificación
graficar_distribucion_clases_1(df)
X_train, X_test, y_train, y_test = particion_estratificada_1(df)
ajustar_arbol_decision_1(X_train, X_test, y_train, y_test)

#12-

import pandas as pd
from clasificacion11y12 import graficar_distribucion_clases_2, particion_estratificada_2, ajustar_arbol_decision_2

# Cargar el DataFrame con los datos procesados
df = pd.read_csv('datos_procesados.csv')

# Utilizar las funciones de clasificación
graficar_distribucion_clases_2(df)
X_train, X_test, y_train, y_test = particion_estratificada_2(df)
ajustar_arbol_decision_2(X_train, X_test, y_train, y_test)