import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import torchaudio
import torchaudio.transforms as T

# ARQUITETURA DO MODELO 
# Esta classe deve ser idêntica à do script de treinamento
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet, self).__init__()
        # Carrega o modelo com os pesos pré-treinados 
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Congela/Descongela as mesmas camadas do treino
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True 

        # Substitui a camada final
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)


# FUNÇÃO DE PREVISÃO 
def classificar_audio(caminho_do_audio, modelo, classes, device):
    
    SR_PADRAO = 22050
    
    try:
        # Carregar com Torchaudio
        # waveform: Tensor de áudio
        # sr: Sample rate original
        waveform, sr = torchaudio.load(caminho_do_audio)
        waveform = waveform.to(device) # Move o áudio para a GPU (se disponível)
        
        # Resample (se necessário) - Torchaudio faz isso fácil
        if sr != SR_PADRAO:
            resampler = T.Resample(orig_freq=sr, new_freq=SR_PADRAO).to(device)
            waveform = resampler(waveform)
            
        # Garante que o áudio seja mono (necessário para o espectrograma)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Criar o Melspectrogram direto para o tensor
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=SR_PADRAO,
            n_mels=224 # Deve ser o mesmo n_mels usado no treino
        ).to(device)
        
        mel_spec = mel_spectrogram_transform(waveform)
        
        # Converter para dB
        # T.AmplitudeToDB() converte para decibéis
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        # Normalizar o espectrograma para 0-1 (para PIL) e converter
        #    (Isso é necessário para "simular" uma imagem antes das transforms)
        mel_spec_db_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Converte para PIL Image (precisa ir para CPU e ter 3 canais)
        # .expand(3, -1, -1) repete o tensor 3x no lugar de Grayscale(3)
        imagem_pil = transforms.ToPILImage()(mel_spec_db_normalized.cpu().expand(3, -1, -1))
        
        # Pré-processamento (Início do pipeline de Visão Computacional)
        IMG_SIZE = 224 
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),      
            
            # ATENÇÃO: Removemos transforms.Grayscale(num_output_channels=3)
            # pois o .expand(3, -1, -1) acima já fez esse trabalho.
            
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
        confianca = confidence.item() * 100

        return previsao, confianca

    except FileNotFoundError:
        return f"Erro: O arquivo de áudio não foi encontrado em '{caminho_do_audio}'", 0
    except Exception as e:
        # Adiciona o 'device' na mensagem de erro se for um problema de processamento
        return f"Ocorreu um erro ao processar o áudio (Dispositivo: {device}): {e}", 0


#execução principal
if __name__ == '__main__':
    
    MODELO_PATH = r'C:\Users\arthur.almeida\Downloads\resnet50_instrumentos.pth' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    if not os.path.exists(MODELO_PATH):
        print(f"ERRO: Checkpoint '{MODELO_PATH}' não encontrado.")
        exit()
    
    try:
        checkpoint = torch.load(MODELO_PATH, map_location=device)
        num_classes = checkpoint['num_classes']
        idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        classes_do_modelo = [idx_to_class[i] for i in range(num_classes)] 

        resnet_carregada = FineTunedResNet(num_classes=num_classes).to(device)
        resnet_carregada.load_state_dict(checkpoint['model_state_dict'])
        resnet_carregada.eval() 

        print(f"Modelo ResNet50 carregado! (Treinado por {checkpoint['epoch']} épocas, Acurácia: {checkpoint['best_val_acc']:.2f}%)")
        print(f"Classes mapeadas ({num_classes}): {classes_do_modelo}")

    except Exception as e:
        print(f"ERRO ao carregar o checkpoint: {e}")
        print("Verifique se a classe 'FineTunedResNet' está definida corretamente e se o checkpoint não está corrompido.")
        exit()

    while True:
        caminho_input = input("\nDIGITE O CAMINHO DO ARQUIVO DE ÁUDIO (ou 'sair'): ").strip('"')
        if caminho_input.lower() == 'sair':
            break

        previsao_instrumento, nivel_confianca = classificar_audio(caminho_input, resnet_carregada, classes_do_modelo, device)
        
        print("-" * 50)
        print(f"Analisando o áudio: {os.path.basename(caminho_input)}")
        if nivel_confianca > 0:
            print(f"--> Instrumento previsto: **{previsao_instrumento.upper()}** ({nivel_confianca:.2f}% de confiança)")
        else:
            print(f"--> {previsao_instrumento}")
        print("-" * 50)
