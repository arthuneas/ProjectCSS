import os
import shutil
import random
import librosa
import librosa.display
import librosa.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
from rich.console import Console
from rich.panel import Panel

warnings.filterwarnings("ignore", category=UserWarning)

console = Console()
PASTA_DESTINO_FINAL = './dataset_instrumentos'
VERSOES_POR_AUDIO = 8
SR_PADRAO = 22050
PROPORCOES = {'train': 0.7, 'validation': 0.2, 'test': 0.1}
FIG_SIZE_CONSISTENTE = (2.24, 2.24) 
CMAP_CONSISTENTE = 'magma'    

LIMPA_TELA = lambda: os.system('cls' if os.name == 'nt' else 'clear')

def gerar_aumentos_audio(y, sr):
    if random.random() < 0.7:
        n_steps = random.uniform(-2.0, 2.0)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    if random.random() < 0.7:
        rate = random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)
    if random.random() < 0.8:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        if np.amax(y) > 0:
            y = y + noise_amp * np.random.normal(size=y.shape)
    if len(y) > sr * 3 and random.random() < 0.6:
        start = random.randint(0, int(len(y) * 0.1))
        end = random.randint(int(len(y) * 0.9), len(y))
        y = y[start:end]
    return y

def salvar_espectrograma_consistente(y, sr, caminho_saida):
    if np.abs(y).max() == 0:
        return
    
    # Limitar o fmax a 8000Hz concentra os 224 bins nas frequências que importam
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224, fmax=8000, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=FIG_SIZE_CONSISTENTE)
    
    librosa.display.specshow(S_db, sr=sr, cmap=CMAP_CONSISTENTE)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(caminho_saida, dpi=100, bbox_inches='tight', pad_inches=0) 
    plt.close()


def processar_audio(caminho_audio, pasta_destino, split):
    try:
        y, sr = librosa.load(caminho_audio, sr=SR_PADRAO)
        nome_base = os.path.splitext(os.path.basename(caminho_audio))[0]
        
        num_versoes = VERSOES_POR_AUDIO if split == 'train' else 1
        
        tamanho_fixo = int(SR_PADRAO * 3.0)
        
        for i in range(num_versoes):
            y_mod = gerar_aumentos_audio(y, sr) if i > 0 else y
            y_mod = librosa.util.fix_length(y_mod, size=tamanho_fixo)
            caminho_saida = os.path.join(pasta_destino, f"{nome_base}_v{i+1}.png")
            salvar_espectrograma_consistente(y_mod, sr, caminho_saida)
    except Exception:
        pass 

if __name__ == '__main__':
    LIMPA_TELA()
    console.print(Panel.fit("\n[bold cyan]GERADOR DE DATASET DE ESPECTROGRAMA \n     (Com Data Aumentation)[/bold cyan]\n", border_style="blue"))
    
    PASTA_ORIGEM_AUDIOS = input("Digite o Diretório de Origem do Dataset IRMAS: ").strip('"')

    if not os.path.exists(PASTA_ORIGEM_AUDIOS):
        LIMPA_TELA()
        console.print("[bold red]ERRO:[/bold red] Diretório de origem não encontrado.")
        exit()

    if os.path.exists(PASTA_DESTINO_FINAL):
        shutil.rmtree(PASTA_DESTINO_FINAL, ignore_errors=True)

    tarefas = []
    contagem_originais = {'train': 0, 'validation': 0, 'test': 0}
    
    classes = [d for d in os.listdir(PASTA_ORIGEM_AUDIOS) if os.path.isdir(os.path.join(PASTA_ORIGEM_AUDIOS, d))]
    
    for nome_classe in classes:
        pasta_classe = os.path.join(PASTA_ORIGEM_AUDIOS, nome_classe)
        arquivos_wav = [f for f in os.listdir(pasta_classe) if f.lower().endswith(".wav")]
        random.shuffle(arquivos_wav) 

        num_arquivos = len(arquivos_wav)
        corte_treino = int(num_arquivos * PROPORCOES['train'])
        corte_valid = corte_treino + int(num_arquivos * PROPORCOES['validation'])

        for i, f in enumerate(arquivos_wav):
            if i < corte_treino:
                split = 'train'
            elif i < corte_valid:
                split = 'validation'
            else:
                split = 'test'
            
            contagem_originais[split] += 1
            pasta_destino = os.path.join(PASTA_DESTINO_FINAL, split, nome_classe)
            os.makedirs(pasta_destino, exist_ok=True)
            tarefas.append((os.path.join(pasta_classe, f), pasta_destino, split))

    num_workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        caminhos = [t[0] for t in tarefas]
        destinos = [t[1] for t in tarefas]
        splits_list = [t[2] for t in tarefas]
        map_iterator = executor.map(processar_audio, caminhos, destinos, splits_list)
        list(tqdm(map_iterator, total=len(tarefas), desc="Gerando Espectrogramas"))
        
    LIMPA_TELA()

    total_imagens = (contagem_originais['train'] * VERSOES_POR_AUDIO) + contagem_originais['validation'] + contagem_originais['test']
    
    status_texto = f"Dataset Finalizado com Sucesso!\n[bold green]Total de PNGs Formadas: {total_imagens}[/bold green]"
    console.print(Panel.fit(status_texto, border_style="green"))

    splits = list(contagem_originais.keys())
    antes = [contagem_originais[s] for s in splits]
    depois = [(contagem_originais[s] * VERSOES_POR_AUDIO if s == 'train' else contagem_originais[s]) for s in splits]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, antes, width, label='Áudios Originais', color="#93ca69")
    rects2 = ax.bar(x + width/2, depois, width, label='Imagens Geradas', color="#9f2757")

    ax.set_ylabel('Quantidade de Arquivos')
    ax.set_title('Comparativo de Arquivos: Antes e Depois do Data Augmentation')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    os.makedirs('./DataImages', exist_ok=True)
    plt.savefig('./DataImages/DistribuiçãoDataset.png', dpi=300)
    plt.close()