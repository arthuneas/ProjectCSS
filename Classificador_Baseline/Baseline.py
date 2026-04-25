import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
LIMPA_TELA = lambda: os.system('cls' if os.name == 'nt' else 'clear')

LIMPA_TELA()
PASTA_ORIGEM = input("Digite o Diretório de Origem do Dataset IRMAS: ").strip()

X_treino, y_treino = [], []
X_teste, y_teste = [], []

def extrair_features(caminho_audio):
    try:    
        y, sr = librosa.load(caminho_audio, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        vetor_1d = np.mean(mfccs, axis=1)
        return vetor_1d
    
    except Exception as e:
        console.print(f"[red]Erro em {caminho_audio}: {e}[/red]")
        return None

classes = [d for d in os.listdir(PASTA_ORIGEM) if os.path.isdir(os.path.join(PASTA_ORIGEM, d))]
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

for cls in classes:
    pasta_classe = os.path.join(PASTA_ORIGEM, cls)
    arquivos = [f for f in os.listdir(pasta_classe) if f.lower().endswith('.wav')]
    arquivos_treino, arquivos_teste = train_test_split(arquivos, test_size=0.2, random_state=42)
    
    console.print(f"[cyan]Processando classe:[/cyan] [bold]{cls}[/bold] ({len(arquivos)} arquivos)")
    for f in arquivos_treino:
        caminho = os.path.join(pasta_classe, f)
        vetor_mcc = extrair_features(caminho)
        if vetor_mcc is not None:
            X_treino.append(vetor_mcc)
            y_treino.append(class_to_idx[cls])

    for f in arquivos_teste:
        caminho = os.path.join(pasta_classe, f)
        vetor_mcc = extrair_features(caminho)
        if vetor_mcc is not None:
            X_teste.append(vetor_mcc)
            y_teste.append(class_to_idx[cls])

LIMPA_TELA()
        
X_treino, y_treino = np.array(X_treino), np.array(y_treino)
X_teste, y_teste = np.array(X_teste), np.array(y_teste)

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_treino, y_treino)

previsoes_teste = modelo_rf.predict(X_teste)
previsoes_treino = modelo_rf.predict(X_treino)
        
acc_treino = accuracy_score(y_treino, previsoes_treino)
acc_teste = accuracy_score(y_teste, previsoes_teste)
f1_global = f1_score(y_teste, previsoes_teste, average='weighted')

metricas_texto = (
    f"Acurácia de Treino: [green]{acc_treino * 100:.2f}%[/green]\n"
    f"Acurácia de Validação: [bold green]{acc_teste * 100:.2f}%[/bold green]\n"
    f"F1-Score Global: [yellow]{f1_global:.4f}[/yellow]"
)
console.print(Panel(metricas_texto, title="[bold white]MÉTRICAS FINAIS DO BASELINE[/bold white]", border_style="blue", expand=False))

cm = confusion_matrix(y_teste, previsoes_teste)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, cmap='Oranges', xticks_rotation=45)
os.makedirs('./DataImages', exist_ok=True)
plt.savefig('./DataImages/matriz_baseline.png')
plt.close()

traducao = {
    'voi': 'Voz',
    'pia': 'Piano',
    'cel': 'Violoncelo',
    'cla': 'Clarinete',
    'flu': 'Flauta',
    'gac': 'Violão Acústico',
    'gel': 'Guitarra Elétrica',
    'org': 'Órgão',
    'sax': 'Saxofone',
    'tru': 'Trompete',
    'vio': 'Violino'
}

LIMPA_TELA()

console.print(Panel.fit(
    "\n[bold magenta]INFERÊNCIA BASELINE \n(Random Forest + MFCC)[/bold magenta]\n[dim]Digite 'sair' para encerrar.[/dim]\n", 
    border_style="magenta"
))

while (True):
    caminho = input("\nDigite o caminho do áudio: ").strip('"').strip()
    
    if caminho.lower() == 'sair':
        break
    
    if not os.path.exists(caminho):
        console.print("[bold red]ERRO:[/bold red] Arquivo não encontrado.")
        continue
      
    try:
        sigla_real = os.path.basename(os.path.dirname(caminho))
        instrumento_real = traducao.get(sigla_real, sigla_real)
        caracteristicas = extrair_features(caminho)
        
        if caracteristicas is not None:
            predicao_numerica = modelo_rf.predict([caracteristicas])[0]
            sigla_prevista = idx_to_class[predicao_numerica]
            instrumento_previsto = traducao.get(sigla_prevista, sigla_prevista)
            
            resultado = f"Real: [white]{instrumento_real}[/white]\nPrevisto: [bold cyan]{instrumento_previsto.upper()}[/bold cyan]"
            console.print(Panel.fit(resultado, title="Resultado da Predição", border_style="cyan"))
        
    except Exception as e:
        console.print(f"[red]Erro ao processar: {e}[/red]")