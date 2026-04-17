import os
import shutil
import random
import librosa #processamento de audio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm #barra de progresso para loops
from concurrent.futures import ProcessPoolExecutor
import torch #biblioteca para deep learning
import torchaudio.transforms as T #transformações de áudio em tensores
import torchvision.transforms as VT #transformações de imagem em tensores, usada para redimensionar os espectrogramas


#configurações gerais
PASTA_ORIGEM_AUDIOS = r'C:\Users\...\IRMAS-TrainingData'
PASTA_SAIDA_IMAGENS = './dataset_instrumentos_otimizado' 
PASTA_DESTINO = './dataset_instrumentos_final'

#divisão do dataset em treino, validação e teste (70%, 15%, 15% respectivamente)
PROPORCOES = {'train': 0.7, 'validation': 0.15, 'test': 0.15}

VERSOES_POR_AUDIO = 7 #quantidade de variações por áudio (data augmentation)
SR_PADRAO = 22050 #taxa de amostragem (frequência de captura do áudio)
N_FFT = 2048 #tamanho da janela da transformada de fourier
HOP_LENGTH = 512 #distancia entre janelas
N_MELS = 224 #resolução vertical do espectrograma (eixo das frequências), deve ser igual ao n_mels usado no treino da ResNet

FIG_SIZE_CONSISTENTE = (2.24, 2.24) # Proporção fixa para todas as imagens
CMAP_CONSISTENTE = 'magma'    

#aumento de dados
#função para gerar variações do áudio original, para que a rede neural não "decore" apenas os arquivos originais e aprenda a generalizar o som
def gerar_aumentos_audio(y, sr):
        """
        Aplica aumentos de dados de forma aleatória e controlada.
        """
        # Pitch shift leve (±2 semitons), muda a nota, tom sem mudar a velocidade
        if random.random() < 0.7:
            n_steps = random.uniform(-2.0, 2.0)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        # Time stretch leve (±15%), muda a velocidade sem mudar o tom
        if random.random() < 0.7:
            rate = random.uniform(0.85, 1.15)
            y = librosa.effects.time_stretch(y, rate=rate)

        # Ruído branco proporcional
        if random.random() < 0.8:
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            # Garante que y não seja somente zeros
            if np.amax(y) > 0:
                y = y + noise_amp * np.random.normal(size=y.shape)

        # Random crop, remove pontsas silenciosas ou partes do início/fim, mas só se o áudio for longo o suficiente, ou seja, maior que 3 segundos (sr * 3 amostras)
        if len(y) > sr * 3 and random.random() < 0.6: # 3 segundos de áudio
            start = random.randint(0, int(len(y) * 0.1))
            end = random.randint(int(len(y) * 0.9), len(y))
            y = y[start:end]

        return y


#transforma o audio em tensor
def salvar_espectrograma_consistente(y, sr, caminho_saida):
    if np.abs(y).max() == 0: #se o áudio for completamente silencioso, então salva um tensor de zeros
        return

    # Converte a onda sonora em um tensor 
    waveform = torch.from_numpy(y).unsqueeze(0)  # (1, N)
    
    #transforma o áudio em um espectrograma de Mel, que é uma representação 2D da energia do som ao longo do tempo e das frequências.
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_spec = mel_spectrogram_transform(waveform)
    
    #converte para decibéis, que é uma escala logarítmica mais próxima da percepção humana de volume
    mel_spec_db = T.AmplitudeToDB()(mel_spec)
    
    #expande para 3 canais (RGB) para ser compatível com as redes de visão computacional, que esperam imagens coloridas
    #visto que o espectrograma é originalmente em escala de cinza (1 canal), o .expand(3, -1, -1) repete o mesmo espectrograma em 3 canais
    tensor_final = mel_spec_db.expand(3, -1, -1)
    
    #garante que todas as imagens tenham o mesmo tamanho (224x224), que é o tamanho esperado pela ResNet
    redimensionador = VT.Resize((224, 224))
    tensor_final = redimensionador(tensor_final)
    
    torch.save(tensor_final, caminho_saida)  #salva como .pt para consistência


def processar_audio(caminho_audio, pasta_destino, num_versoes, indice):
    #Processa um único arquivo de áudio gerando múltiplos espectrogramas.
    #aqui ocorre o processo de data augmentation
    try:
        #carrega o sinal de audio (y) e a taxa de amostragem (sr)
        y, sr = librosa.load(caminho_audio, sr=SR_PADRAO)
        nome_base = os.path.splitext(os.path.basename(caminho_audio))[0]

        for i in range(num_versoes):
            #a primeira vrsão é salva da maneira original e o restante passa por distorções para criar variações do mesmo áudio
            # v1 (i=0) é o original, os outros(i > 0) são aumentados
            y_mod = gerar_aumentos_audio(y, sr) if i > 0 else y 
            
            # usa o índice para garantir nomes únicos mesmo com múltiplos arquivos iguais
            caminho_saida = os.path.join(pasta_destino, f"{nome_base}_v{indice + i}.pt")
            
            #chama a função que transforma o áudio em um espectrograma consistente (mesmo tamanho, mesma escala de cores) e salva como .pt
            salvar_espectrograma_consistente(y_mod, sr, caminho_saida) 

    except Exception as e:
        #uma exception para o corrompimento de arquivo, o arquivo avisa o erro e passa para o próximo
        #evitando a paralisação de todo o processo
        print(f"[ERRO] {caminho_audio}: {e}")
      
      
      
      
def gerar_graficos_comparativos(contagem_antes, contagem_depois): 
    #função para gerar gráfico de barrar comparativos entre a quantidade de amostras por classe antes e depois do aumento de dados, 
    #para verificar se o balanceamento foi executado corretamente      
    classes = list(contagem_antes.keys())
    antes = [contagem_antes[cls] for cls in classes]
    depois = [contagem_depois[cls] for cls in classes]
    
    x= np.arange(len(classes)) #localiza as classes no eixo x
    width = 0.35 #largura das barras
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, antes, width, label='Original (WAV)', color='skyblue')
    ax.bar(x + width/2, depois, width, label='Aumentado (PT)', color='lightcoral')
    
    ax.set_ylabel('Quantidade de Amostras')
    ax.set_title('Distribuição de Amostras por Classe Antes e Depois do Aumento')
    plt.xticks(x, classes, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('comparativo_distribuicao_classes.png') #salva o gráfico
    plt.show()




def fase1_gerar_todas_as_imagens():
    #organiza o mapeamento de arquivos, calcula o balanceamento, distribui os trabalhos para o hardware (núcleo do processador)
   
   #verificação de existencia da pasta de origem dos áudios
    if not os.path.exists(PASTA_ORIGEM_AUDIOS):
        print(f"ERRO: Pasta '{PASTA_ORIGEM_AUDIOS}' não encontrada!")
        exit()
    
    #mapeia quais os nomes da classes e a lista de arquivos .wav de cada    
    classes = [d for d in os.listdir(PASTA_ORIGEM_AUDIOS) if os.path.isdir(os.path.join(PASTA_ORIGEM_AUDIOS, d))]
    arquivos_por_classe = {}
    
    for c in classes:
        pasta_classe = os.path.join(PASTA_ORIGEM_AUDIOS, c)
        arquivos = [f for f in os.listdir(pasta_classe) if f.lower().endswith('.wav')]
        arquivos_por_classe[c] = arquivos

    #balanceamento das classes    
    # Coleta todos os caminhos de áudio e prepara destinos
    #conta quantos arquivos tem cada classe originalmente
    contagem_antes = {cls: len(arquivos_por_classe[cls]) for cls in classes}
    
    #analisa a classe com maior quantidade de dados e define que todas devem ter a mesma quantidade
    #multiplica essa quantidade pela quantidade de variações definidas nas configurações
    max_audios = max(contagem_antes.values())
    alvo_por_classe = max_audios * VERSOES_POR_AUDIO
    contagem_depois = {cls: alvo_por_classe for cls in classes}
    
    tarefas = [] #lista que guarda o argumento para cada execução paralela
    
    for c in classes:
        num_arquivos = len(arquivos_por_classe[c])
        pasta_destino = os.path.join(PASTA_SAIDA_IMAGENS, c)
        os.makedirs(pasta_destino, exist_ok=True)
    
        #calcula quantas variações cada arquivo deve precisa gerar para que a classe atinja a quantidade definido (alvo_por_classe)
        versoes_base = alvo_por_classe // num_arquivos 
        resto = alvo_por_classe % num_arquivos
        
        indice_global = 1
        for i, f in enumerate(arquivos_por_classe[c]):
            caminho_audio = os.path.join(PASTA_ORIGEM_AUDIOS, c, f)
            #se houver resto de divisão, distribui uma variação extra para os primeiros arquivos, garantindo que o total final seja exatamente o alvo_por_classe
            num_versoes = versoes_base + (1 if i < resto else 0) 
            tarefas.append((caminho_audio, pasta_destino, num_versoes, indice_global))
            indice_global += num_versoes    


    # Desempacota a lista de tarefas em listas de argumentos separadas
    # prepara as lista de argumentos 
    caminhos_audios = [t[0] for t in tarefas]
    pastas_destino = [t[1] for t in tarefas]
    versoes = [t[2] for t in tarefas]
    indices = [t[3] for t in tarefas]
    
    #define o número de núcleos a serem usados, deixando 2 livres para não travar o sistema
    num_workers = max(1, os.cpu_count() - 2)
    print(f"Quantidade de núcleos: {num_workers}")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        # Agora passamos a função e as listas de argumentos separadamente
        # O map vai executar: processar_audio(caminhos_audios[0], pastas_destino[0])
        map_iterator = executor.map(processar_audio, caminhos_audios, pastas_destino, versoes, indices)
        
        #mostra a barra de progresso e tempo real
        list(tqdm(map_iterator,
                  total=len(tarefas),
                  desc="Gerando espectrogramas"))

    #gera o gráfico para validar o balanceamento realizado
    gerar_graficos_comparativos(contagem_antes, contagem_depois)    
    print(f"\nDataset salvo em '{PASTA_SAIDA_IMAGENS}'\n")



# 
def dividir_arquivos_de_uma_classe(nome_classe, pasta_origem, pasta_destino, proporcoes):
    #Pega todos os arquivos de uma única classe, embaralha e copia para as
    #pastas de treino, validação e teste de destino.
    caminho_completo_origem = os.path.join(pasta_origem, nome_classe)
    
    #coleta os arquivos .pt, os tensores, 
    arquivos = [f for f in os.listdir(caminho_completo_origem) if f.endswith('.pt')]
    
    #shuffle funciona para as versoões originais e aumentadas sejam distribuidas aleatoriamente entre treino e teste
    random.shuffle(arquivos)
    
    #calculo os pontos de corte baseado nas proporções
    total_arquivos = len(arquivos)
    ponto_corte_treino = int(total_arquivos * proporcoes['train'])
    ponto_corte_valid = ponto_corte_treino + int(total_arquivos * proporcoes['validation'])

    conjuntos = {
        'train': arquivos[:ponto_corte_treino],
        'validation': arquivos[ponto_corte_treino:ponto_corte_valid],
        'test': arquivos[ponto_corte_valid:]
    }
    
    #cria as subpastas e copia os arquivos
    for nome_conjunto, lista_arquivos in conjuntos.items():
        pasta_destino_final = os.path.join(pasta_destino, nome_conjunto, nome_classe)
        os.makedirs(pasta_destino_final, exist_ok=True)
        
        for arquivo in lista_arquivos:
            #utilizado para mandder os arquivos para a pasta de destino, mantendo a integridade da pasta 'otimizado' original
            shutil.copy(
                os.path.join(caminho_completo_origem, arquivo),
                pasta_destino_final
            )

def fase2_dividir_train_val_test():
    #gerencia a criação do dataset final estruturado em pastas de treino, validação e teste, 
    #organizando os arquivos .pt gerados na fase 1 para serem usados no treinamento da rede neural
    print("--- Iniciando a divisão do dataset ---")
    
    #limpa a pasta de destino caso ela já exista, para evitar arquivos antigos misturados com os novos
    if os.path.exists(PASTA_DESTINO):
        shutil.rmtree(PASTA_DESTINO)
        print(f"Pasta de destino antiga '{PASTA_DESTINO}' removida.")
    
    try:
        #verifica se a pasta de origem existe e coleta os nomes das classes, que são as pastas dentro da pasta de origem
        classes = [d for d in os.listdir(PASTA_SAIDA_IMAGENS) if os.path.isdir(os.path.join(PASTA_SAIDA_IMAGENS, d))]
        if not classes:
            print(f"ERRO: Nenhuma pasta de classe encontrada em '{PASTA_SAIDA_IMAGENS}'.")
            return
        print(f"Classes detectadas: {', '.join(classes)}")
    except FileNotFoundError:
        print(f"ERRO: Pasta de origem '{PASTA_SAIDA_IMAGENS}' não encontrada.")
        return

    #itera sobre cada classe e chama a função para dividir os arquivos de cada classe em treino, validação e teste
    for nome_classe in tqdm(classes, desc="Processando Classes"):
        dividir_arquivos_de_uma_classe(nome_classe, PASTA_SAIDA_IMAGENS, PASTA_DESTINO, PROPORCOES)

    print(f"\n--- Divisão concluída com sucesso! ---")
    print(f"Dataset final pronto em: '{PASTA_DESTINO}'")


#bloco principal
if __name__ == '__main__':
    print("=== INICIANDO PIPELINE DE DADOS ===")
    
    print("\nGerando Espectrogramas")
    fase1_gerar_todas_as_imagens() #cria os tesnors 224x224 a partir dos áudios, com data augmentation
    
    print("\nDividindo imagens")
    fase2_dividir_train_val_test() #organiza os arquivos .pt para a rede neural
    
    print("\n=== PIPELINE CONCLUÍDO COM SUCESSO! ===")