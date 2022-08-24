import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

ds_prestamos = pd.read_excel('prestamos_bancarios_alemanes_1994.xls')

# quitamos la última fila
ds_prestamos = ds_prestamos.drop([len(ds_prestamos)-1])
# indicamos los atributos explicativos a utilizar
X = ds_prestamos[['Account Balance','Length of current employment','Instalment per cent','Duration in Current address', 'Most valuable available asset', 'Telephone']]
y = ds_prestamos['Creditability']
# separamos los conjunto de prueba y de entrenamiento
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size= 0.30, random_state=0)
classifier = GaussianNB()
# realizamos las pruebas
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

# ingresamos un nuevo solicitante al clasificador
n = {'Account Balance': [2],'Length of current employment':[2],'Instalment per cent':[4],'Duration in Current address':[1], 'Most valuable available asset':[4], 'Telephone':[1]}
ns = pd.DataFrame(data=n)
ns_pred = classifier.predict(ns)
print(ns_pred)
