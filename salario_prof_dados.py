import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados 
data = pd.read_csv('/kaggle/input/state-of-data-2022/State_of_data_2022.csv')

# Dividir os dados em características (X) e variável alvo (y)
X = data[['idade', 'atuacao', 'nivel_formacao', 'experiencia', 'porte_empresa']]
y = data['salario']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Aplicativo Streamlit
st.title('Previsão de Salário')
st.write('Insira os valores abaixo para prever o salário:')

idade = st.number_input('Idade', min_value=0, max_value=100)
atuacao = st.selectbox('Atuação', ['opção1', 'opção2', 'opção3'])  # Substitua com suas opções
nivel_formacao = st.selectbox('Nível de Formação', ['ensino médio', 'graduação', 'pós-graduação'])
experiencia = st.number_input('Experiência (anos)', min_value=0, max_value=50)
porte_empresa = st.selectbox('Porte da Empresa', ['pequena', 'média', 'grande'])

# Prever o salário com base nos valores inseridos
input_data = [[idade, atuacao, nivel_formacao, experiencia, porte_empresa]]
predicted_salary = model.predict(input_data)

st.write(f'Previsão de Salário: R${predicted_salary[0]:.2f}')
