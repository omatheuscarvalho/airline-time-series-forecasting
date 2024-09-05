# Problem Context

I am currently taking a class on Time Series, which is an introductory course covering the basic concepts of machine learning. It's an elective course, so it's not part of the core curriculum. There are no exams, only assignments where we are tasked with researching, implementing, and presenting solutions.

For one of our assignments, the professor has asked us to select a dataset from Kaggle and implement prediction models using both ARIMA and MLP. Afterward, we are required to calculate the WMAPE (Weighted Mean Absolute Percentage Error) to evaluate which model performs better.

# Dataset Description

I chose this dataset, which contains monthly totals of international airline passengers from 1949 to 1960. The dataset includes two columns: the first represents the date, and the second represents the number of passengers. This dataset is often used for time series analysis and forecasting, making it an ideal choice for implementing models like ARIMA and MLP. The clear trend and seasonality patterns present in the data provide a great opportunity to evaluate the performance of these prediction models.

# 1. Implementing ARIMA

Eu comcei com o básico, adicionando o dataset e lendo ele com pandas. Posteriormente, plotei o gráfico para entender como ele estava se comportando

```python
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, index_col='Month', parse_dates=True)
plt.plot(data)
plt.title('Air passagens over time')
plt.xlabel('Time (mounths)')
plt.ylabel('Passangers')
plt.show()
```

![First Print](./images/arima-01/saz-print.png)
Depois, fiz algumas pesquisas e análises para tentanr entender o que poderia ser extraído de informações desse gráfico, e cheguei nessa conclusão simples:

- Tendência: crescimento ao longo dos anos
- Sazionalidade: existe, aparentermente em períodos parecidos, o que pode indicar que existem épocas específicas de aumento (ferás, feriados, etc)
- Volatilidade Crescente: a variação entre os meses tamém está sendo maior

Então fui para separação dos dados, 80% para treino e 20% para testes

```python
train_size_sazionalido = int(len(data) * 0.8)
train_data_sazionalizado = data[:train_size_sazionalido]
test_data_sazionalizado = data[train_size_sazionalido:]

#Plotando codigo
plt.plot(train_data_sazionalizado, label='Treino')
plt.plot(test_data_sazionalizado, label='Teste')
plt.title('Air passagens over time')
plt.xlabel('Time (mounths)')
plt.ylabel('Passangers')
plt.legend()
```

![Test and train](./images/arima-01/test-train.png)

Excelente, até aqui tudo ocorreu bem, então resolvi ajustar o modelo ARIMa, pelas configurações que encontrei, essa pareceu ser uma adequada:

```python
model = ARIMA(train_data_sazionalizado, order=(5,1,0))
model_fit = model.fit()
```

E então, hora de fazer a previsão e plotar o gráfico:

```python
arima_pred = model_fit.forecast(steps=len(test_data_sazionalizado))

plt.plot(train_data_sazionalizado, label='Treino')
plt.plot(test_data_sazionalizado, label='Teste')
plt.plot(arima_pred, label='Previsão', linestyle='--')
plt.title('Air passagens over time')
plt.xlabel('Time (mounths)')
plt.ylabel('Passangers')
plt.legend()
```

![Sazional plot](./images/arima-01/pred-1.png)
Batendo o olho no gráfico eu ja tinha entendi que ele não podia estar certo, então lembrei que a professora tinha falado sobre sazionalidade e que precisariamos remove-la, então refiz o gráfico a fim de torna-lo desasazional:

```python
result_sazional = seasonal_decompose(train_data_sazionalizado, model='additive', period=12)

plt.plot(result_sazional.trend, label='Deseasonalized Trend')
plt.title('Air Passengers Deseasonalized Trend')
plt.xlabel('Time (months)')
plt.ylabel('Passengers')
plt.legend()
plt.show()
```

![Desazional data](./images/arima-01/desazional-data.png)

Então refiz todo o processo e cheguei no grafico a seguir

```python
train_size = int(len(result_sazional.trend) * 0.8)
train_deseasonalized = result_sazional.trend[:train_size]
test_deseasonalized = result_sazional.trend[train_size:]
model_deseasonalized = ARIMA(train_deseasonalized, order=(5,1,0))
model_deseasonalized_fit = model_deseasonalized.fit()
arima_pred_deseasonalized = model_deseasonalized_fit.forecast(steps=len(test_deseasonalized))
plt.plot(train_deseasonalized, label='Deseasonalized Train')
plt.plot(test_deseasonalized, label='Deseasonalized Test')
plt.plot(arima_pred_deseasonalized, label='Deseasonalized ARIMA Prediction', linestyle='--')
plt.title('Air Passengers Deseasonalized Trend and ARIMA Prediction')
plt.xlabel('Time (months)')
plt.ylabel('Passengers')
plt.legend()
plt.show()
```

![Arima 80 DS](./images/arima-02-ds/80.png)

Então, apliquei o teste o WMAPE tanto para o Sazionalizado quando para o Desazionalizado

```python
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred) / y_true) / len(y_true)

wape_arima = wmape(test_data_sazionalizado['Passengers'].values, arima_pred)
print(f'WMAPE do modelo ARIMA Sazionalizado: {wape_arima * 100:.2f}%')

wape_arima_deseasonalized = wmape(test_deseasonalized.values, arima_pred_deseasonalized)
print(f'WMAPE do modelo ARIMA desazonalizado: {wape_arima_deseasonalized * 100:.2f}%')
```

E obtive os seguintes resultados:  
WMAPE do modelo ARIMA Sazionalizado: 16.15%  
WMAPE do modelo ARIMA desazonalizado: 1.85%

Depois disso, fiz mais alguns testes com outras proporções e cheguei nesses resultados:  
(Considere as porcentagens representadas no eixo x como % alocada para o modelo de treino)
![Tests Arima-ds](./images/arima-02-ds/testes.png)

## Como poderia melhorar o resultado do WMAPE?

Fiquei me questionando em alterar os parâmetros para tentar encontrar um valor ainda menor, então resolvi tentar todas as possibilidades de combinações, pedi uma recomendação de intervalo para p, d e q para o chat gpt e ele me retornou com 0-6 para p e q, 0-2 para d.  
Então montei o loop para testar todas as possibilidades dessas parâmetros associados as porcentagem de treino de 0.75....0.85 junto com 0.75 e 0.90. Por fim, ela iria salvar todos os resultados em um arquivo csv que está disponível também nesse git.

```python
# Função WMAPE
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

# Definir o range de valores para p, d, q e proporção de treino
p_values = range(0, 6)
d_values = range(0, 2)
q_values = range(0, 6)

# Testar diferentes proporções de treino: 50%, 60%, 70%, 80%, 85%
train_proportions = [0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.9]

# Criar todas as combinações possíveis de p, d, q
pdq_combinations = list(itertools.product(p_values, d_values, q_values))

# Lista para armazenar os resultados
results = []

# Loop para testar cada proporção de treino e cada combinação de p, d, q
for train_prop in train_proportions:
    # Definir o tamanho do conjunto de treino com base na proporção atual
    train_size = int(len(result_sazional.trend) * train_prop)

    # Dividir os dados em treino e teste com base na proporção atual
    train_deseasonalized = result_sazional.trend[:train_size]
    test_deseasonalized = result_sazional.trend[train_size:]

    # Testar cada combinação de p, d, q
    for pdq in pdq_combinations:
        try:
            # Ajustar o modelo ARIMA para cada combinação de p, d, q
            model = ARIMA(train_deseasonalized, order=pdq)
            model_fit = model.fit()

            # Fazer previsões
            predictions = model_fit.forecast(steps=len(test_deseasonalized))

            # Calcular WMAPE
            wmape_value = wmape(test_deseasonalized, predictions)

            # Armazenar os resultados em um dicionário
            result = {
                'p': pdq[0],
                'd': pdq[1],
                'q': pdq[2],
                'train_proportion': train_prop,
                'wmape': wmape_value
            }
            results.append(result)
        except:
            continue

# Converter os resultados em um DataFrame
df_results = pd.DataFrame(results)

# Exibir o DataFrame com os resultados
display(df_results)
# Salvar os resultados em um arquivo CSV para análise posterior
df_results.to_csv('arima_wmape_results.csv', index=False)
```

Disclaimer => Posteriormente descobri que o que eu fiz se chama GridSearch e já existem métodos prontos para faze-lo.

A melhor combinação resultante foi:
Melhor combinação de parâmetros:  
p = 0.0  
d = 1.0  
q = 5.0  
train_proportion = 90%  
wmape = 0.178179

Outro dado interessante também foi o wape relacionado a proporção de treino:
![Wape/train proportion](./images/arima-02-ds/wape-train-proportion.png)

Essa relação foi importante, pois o que eu achava que era o mais interessante, no caso 78, na verdade poderia não ser.

Levantei outros dados relacionais também, mas além de plotar um gráfico juntando os "melhores" resoltudados, não cheguei a nenhuma conclusão relevante.
![Wape/p](./images/arima-02-ds/wape-per-p.png)
![Wape/d](./images/arima-02-ds/wape-per-d.png)
![Wape/q](./images/arima-02-ds/wape-per-q.png)

E abaixo as melhores combinações encontradas pelo algoritimo:
![best combinations](./images/arima-02-ds/best-proportions.PNG)

Em seguida, plotei as duas melhores combinações:
![best 2 comb](./images/arima-02-ds/best-2-comb.PNG)

## Adicionando parâmetros

Estava tendo algumas dificuldades com o código do LMP e resolvi pedir ajuda a um colega e ele me sugeriu algumas alterações.  
Adicionando o parâmetro de sazional_order e fazendo a previsão, tivemos uma boa melhora do gráfico sazional com o modelo arima:

```python
model_so = ARIMA(train_data_sazionalizado, seasonal_order=(12,1,0,12))
model_so_fit = model_so.fit()

arima_pred_so = model_so_fit.forecast(steps=len(test_data_sazionalizado))
```

![sazional-parameter](./images/arime-03/predition.png)
