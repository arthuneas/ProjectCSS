import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import librosa
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import librosa.display
from rich.console import Console
from rich.panel import Panel

console = Console()
LIMPA_TELA = lambda: os.system('cls' if os.name == 'nt' else 'clear')

class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
       
        for param in self.resnet.parameters():
            param.requires_grad = False

        for name, param in self.resnet.named_parameters():
            if "layer4" in name or "layer3" in name or "fc" in name:
                param.requires_grad = True

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)

def classificar_audio(caminho_do_audio, modelo, classes, device):
    SR_PADRAO = 22050
    FIG_SIZE_CONSISTENTE = (2.24, 2.24)
    CMAP_CONSISTENTE = 'magma'
   
    try:
        y, sr = librosa.load(caminho_do_audio, sr=SR_PADRAO)
       
        if np.abs(y).max() == 0:
            return "Áudio silencioso", 0

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=FIG_SIZE_CONSISTENTE)
        librosa.display.specshow(S_db, sr=sr, cmap=CMAP_CONSISTENTE)
        plt.axis('off')
        plt.tight_layout(pad=0)
       
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
       
        imagem_pil = Image.open(buf).convert('RGB')
        buf.close()

        IMG_SIZE = 224
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),          
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
       
        imagem_tensor = transform(imagem_pil).unsqueeze(0).to(device)

        modelo.eval()
        with torch.no_grad():
            output = modelo(imagem_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
       
        previsao = classes[predicted_idx.item()]

        return previsao

    except FileNotFoundError:
        return f"Erro: Arquivo não encontrado em '{caminho_do_audio}'", 0
    except Exception as e:
        return f"Erro no processamento: {e}", 0

if __name__ == '__main__':
    LIMPA_TELA()
    
    TRADUCAO_INSTRUMENTOS = {
    'cel': 'Violoncelo',
    'cla': 'Clarinete',
    'flu': 'Flauta',
    'gac': 'Violão Acústico', 
    'gel': 'Guitarra Elétrica', 
    'org': 'Órgão',
    'pia': 'Piano', 
    'sax': 'Saxofone', 
    'tru': 'Trompete',
    'vio': 'Violino', 
    'voi': 'Voz'
}
    
    MODELO_PATH = r'/Users/apple/Downloads/resnet50_instrumentos.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODELO_PATH):
        console.print(Panel(f"[bold red]ERRO:[/bold red] Checkpoint não encontrado em:\n{MODELO_PATH}", border_style="red"))
        exit()
        
    LIMPA_TELA()
   
    try:
        checkpoint = torch.load(MODELO_PATH, map_location=device)
        num_classes = checkpoint['num_classes']
        idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        classes_do_modelo = [idx_to_class[i] for i in range(num_classes)]

        resnet_carregada = FineTunedResNet(num_classes=num_classes).to(device)
        resnet_carregada.load_state_dict(checkpoint['model_state_dict'])
        resnet_carregada.eval()

        info_modelo = (
            f"Dispositivo: [bold cyan]{device}[/bold cyan]\n"
            f"Épocas Treinadas: [yellow]{checkpoint['epoch']}[/yellow]\n"
            f"Acurácia do Checkpoint: [green]{checkpoint['best_val_acc']:.2f}%[/green]"
        )
        console.print(Panel(info_modelo, title="[bold white]MODELO RESNET50[/bold white]", border_style="blue", expand=False))

    except Exception as e:
        console.print(f"[bold red]ERRO ao carregar o checkpoint:[/bold red] {e}")
        exit()

    console.print(Panel.fit(
        "\n[bold cyan]INFERÊNCIA DEEP LEARNING \n(ResNet50 + Espectrogramas)[/bold cyan]\n[dim]Digite 'sair' para encerrar.[/dim]\n", 
        border_style="cyan"
    ))

    while True:
        caminho = input("\nDigite o caminho do áudio: ").strip('"').strip()
        
        if caminho.lower() == 'sair':
            break

        previsao_sigla, nivel_confianca = classificar_audio(caminho, resnet_carregada, classes_do_modelo, device)
        previsao_instrumento = TRADUCAO_INSTRUMENTOS.get(previsao_sigla, previsao_sigla)
        
        if nivel_confianca > 0:
            nome_arquivo = os.path.basename(caminho)
            match = re.search(r'\[([a-z]{3})\]', nome_arquivo)
            
            if match:
                sigla_real = match.group(1)
            else:
                pasta_pai = os.path.basename(os.path.dirname(caminho))
                if pasta_pai in TRADUCAO_INSTRUMENTOS:
                    sigla_real = pasta_pai
                else:
                    sigla_real = "desconhecido"
            
            instrumento_real = TRADUCAO_INSTRUMENTOS.get(sigla_real, "Não identificado")
            
            if sigla_real == "desconhecido":
                cor_previsao = "yellow"
                texto_real = "[dim]Não identificado no nome/pasta[/dim]"
            else:
                cor_previsao = "green" if previsao_instrumento.upper() == instrumento_real.upper() else "red"
                texto_real = f"[bold cyan]{instrumento_real.upper()}[/bold cyan]"

            resultado_texto = (
                f"Instrumento Real: {texto_real}\n"
                f"Previsão do Modelo: [bold {cor_previsao}]{previsao_instrumento.upper()}[/bold {cor_previsao}]\n"
            )
            
            console.print(Panel.fit(
                resultado_texto, 
                title=f"{nome_arquivo}", 
                border_style=cor_previsao
            ))
        else:
            console.print(f"[bold red]AVISO:[/bold red] {previsao_instrumento}")