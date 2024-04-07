import pandas as pd

# Veri setini yükleme
file_path = "/Users/oguzkaanyalcin/Desktop/piton/database.txt"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(file_path, names=names, delimiter='\t')  # Metin dosyası olduğu için delimiter belirtmek gerekebilir

# Veri setini özellikler (X) ve hedef değişken (y) olarak ayırma
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Veri setini %70 eğitim ve %30 test olarak ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizasyon işlemi
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA işlemi
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# LDA işlemi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# Çoklu Doğrusal Regresyon analizi
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)
mlr_coefficients = mlr.coef_

# Multinominal Lojistik Regresyon analizi
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
log_reg.fit(X_train_scaled, y_train)
log_reg_coefficients = log_reg.coef_

# Karar Ağacı Sınıflandırma
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_scaled, y_train)
dt_predictions = dt_classifier.predict(X_test_scaled)

# Naive Bayes Sınıflandırıcısı
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)
nb_predictions = nb_classifier.predict(X_test_scaled)










from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Çoklu Doğrusal Regresyon için performans metrikleri
mlr_predictions = mlr.predict(X_test_scaled)
mlr_conf_matrix = confusion_matrix(y_test, mlr_predictions)
mlr_accuracy = accuracy_score(y_test, mlr_predictions)
mlr_precision = precision_score(y_test, mlr_predictions)
mlr_recall = recall_score(y_test, mlr_predictions)
mlr_f1 = f1_score(y_test, mlr_predictions)

# Multinominal Lojistik Regresyon için performans metrikleri
log_reg_predictions = log_reg.predict(X_test_scaled)
log_reg_conf_matrix = confusion_matrix(y_test, log_reg_predictions)
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
log_reg_precision = precision_score(y_test, log_reg_predictions)
log_reg_recall = recall_score(y_test, log_reg_predictions)
log_reg_f1 = f1_score(y_test, log_reg_predictions)

# Karar Ağacı Sınıflandırma için performans metrikleri
dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)

# Naive Bayes Sınıflandırıcısı için performans metrikleri
nb_conf_matrix = confusion_matrix(y_test, nb_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)

# Performans metriklerini yazdırma
print("Çoklu Doğrusal Regresyon Performans Metrikleri:")
print("Confusion Matrix:\n", mlr_conf_matrix)
print("Accuracy:", mlr_accuracy)
print("Precision:", mlr_precision)
print("Recall:", mlr_recall)
print("F1 Score:", mlr_f1)
print("\n")

print("Multinominal Lojistik Regresyon Performans Metrikleri:")
print("Confusion Matrix:\n", log_reg_conf_matrix)
print("Accuracy:", log_reg_accuracy)
print("Precision:", log_reg_precision)
print("Recall:", log_reg_recall)
print("F1 Score:", log_reg_f1)
print("\n")

print("Karar Ağacı Sınıflandırma Performans Metrikleri:")
print("Confusion Matrix:\n", dt_conf_matrix)
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
print("\n")

print("Naive Bayes Sınıflandırıcısı Performans Metrikleri:")
print("Confusion Matrix:\n", nb_conf_matrix)
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)

