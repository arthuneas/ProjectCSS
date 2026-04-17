import os
import librosa #biblioteca para processamento de áudio
import numpy as np #biblioteca para manipulação de matrizes e cálculos numéricos
from sklearn.ensemble import RandomForestClassifier #modelo de classificação
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay #métricas de avaliação
from sklearn.model_selection import train_test_split #para dividir os dados em treino e teste
import matplotlib.pyplot as plt #para visualização da matriz de confusão e desenho de gráficos


PASTA_ORIGEM = r'C:\Users\arthur.almeida\Downloads\IRMAS-TrainingData'

#X sâo as features, MFCCs, características extraídas dos áudios
#y sâo as labels, ou seja, os instrumentos correspondentes a cada áudio
X_treino, y_treino = [], []
X_teste, y_teste = [], []


def extrair_features(caminho_audio):
    #Primeiro: Carrega o áudio puro
    #y é a onda sonora, um vetor de amplitude ao longo do tempo
    #sr é a taxa de amostragem, ou seja, quantas vezes por segundo o áudio foi amostrado (ex: 22050 Hz)
    y, sr = librosa.load(caminho_audio, sr=22050)
    
    #Segundo: Extrai a matriz de MFCCs, nesse caso, 40 coeficientes
    #o mfcc é uma representação matemática que tenta imitar como o ouvido humano percebe as frequências
    #o resultado é uma matriz 2D: 40 coeficientes no eixo Y, e vários frames de tempo no eixo X
    #ao tirar a media no axis=1 (eixo do tempo), esmagamos a variação temporal, o resultado vira um vetor 1D simples com 40 números.
    #representa a média do som inteiro, o que é necessário para o modelo, que precisa de um vetor de tamanho fixo
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    #Terceiro: Tira a média ao longo do eixo do tempo (axis=1)
    #o resultado vira um vetor 1D simples com 40 números, a média do som inteiro, o que é necessário o modelo, que precisa de um vetor de tamanho fixo
    #
    vetor_1d = np.mean(mfccs, axis=1)
    
    return vetor_1d


#Pega o nome das pastas, que são as classes dos instrumentos (ex: 'cel', 'cla', 'flu')
classes = [d for d in os.listdir(PASTA_ORIGEM) if os.path.isdir(os.path.join(PASTA_ORIGEM, d))]

#cria um dicionário para converter nome de texto em número (ex: {'cel': 0, 'cla': 1})
#modelos matemáticos só entendem números, não palavras
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}


#o loop intera sobre cada classe de instrumento
for cls in classes:
    pasta_classe = os.path.join(PASTA_ORIGEM, cls)
    arquivos = [f for f in os.listdir(pasta_classe) if f.lower().endswith('.wav')]
    
    #primeiro divide os arquivos desta classe: 80% treino, 20% teste, pelo test_size=0.2
    #isso garante que teremos uma amostra balanceada, ou seja, se tivermos 100 áudios na classe A e 1000 na B, 
    #garantimos que teremos de proporção para ambas, sem desbalancear o treino
    arquivos_treino, arquivos_teste = train_test_split(arquivos, test_size=0.2, random_state=42)
    
    #Processamento do treino
    print(f"Extraindo features de TREINO da classe: {cls}...")
    for f in arquivos_treino:
        caminho = os.path.join(pasta_classe, f) #monta o caminho completo do arquivo de áudio
        
        vetor_mcc = extrair_features(caminho) #usa a função para transformar o áudio em um vetor de características (MFCCs)
        X_treino.append(vetor_mcc) #armazena o vetor de características na lista de treino
        y_treino.append(class_to_idx[cls]) #armazena a label correspondente (número do instrumento) na lista de treino

                
    #Processamento do teste
    #o processo é o mesmo do treino, mas armazena nas listas de teste
    print(f"Extraindo features de TESTE da classe: {cls}...")
    for f in arquivos_teste:
        caminho = os.path.join(pasta_classe, f)
        
        vetor_mcc = extrair_features(caminho)
        X_teste.append(vetor_mcc)
        y_teste.append(class_to_idx[cls])
        
        
#converte as listas de treino e teste para arrays numpy, que são o formato esperado pelos modelos de machine learning
X_treino = np.array(X_treino)
y_treino = np.array(y_treino)
X_teste = np.array(X_teste)
y_teste = np.array(y_teste)


#1 -instancia o modelo clássico
#'n_estimators=100' significa que 100 árvores de decisão serõ criadas, onde cada arvore é treinada em uma amostra aleatória dos dados e faz previsões, 
#e o resultado final é a média das previsões de todas as árvores
#'n_jobs=-1' é um parâmetro para informar ao hardware o uso de todos os núcleos para o treinamento, acelerando o processo.
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


#2 - treina o modelo com os dados de treino
#o modelo compara os dados de x_treino (características) com y_treino (labels) para aprender a associar as características aos instrumentos correspondentes
modelo_rf.fit(X_treino, y_treino)


#3 - Prevê os dados separados para teste
#pegamos dados que o modelo nunca viu antes (X_teste) e pedimos para ele prever quais instrumentos são (y_teste)
previsoes = modelo_rf.predict(X_teste)
        

#4 - Avaliação
#Calcule e imprima a Acurácia, porcentagem de previsões corretas
acc = accuracy_score(y_teste, previsoes)
#calculo do f1-score, uma razão entre precisão(quantidade de previsões corretas por total de previsões) e recall (verdadeiros positivos por total de casos reais)
f1 = f1_score(y_teste, previsoes, average='weighted')


#5 - Plota a Matriz de Confusão 
#assim, conseguimos visualizar onde o modelo errou, ou seja, quais instrumentos ele confundiu mais frequentemente
cm = confusion_matrix(y_teste, previsoes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Oranges', xticks_rotation=45)
plt.title('Matriz de Confusão - Baseline Random Forest (MFCC)')
plt.tight_layout()
plt.savefig('./matriz_baseline.png')
print(f"\nResultados do Baseline:\nAcurácia: {acc:.4f}\nF1-Score: {f1:.4f}")