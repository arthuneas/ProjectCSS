import os
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# CONFIGURAÇÕES E PRÉ-PROCESSAMENTO
# Define os hiperparâmetros e configurações do processamento.
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 2 
MODEL_SAVE_PATH = './model_checkpoint/resnet50_instrumentos.pth'

# Define a sequência de transformações para os dados de treino, incluindo data augmentation.
transform_treino = transforms.Compose([ 
    T.FrequencyMasking(freq_mask_param = 30),
    T.TimeMasking(time_mask_param = 50),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a sequência de transformações para os dados de teste.
transform_teste = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ARQUITETURA DO MODELO
# Define a classe do modelo de classificação baseado na ResNet.
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet, self).__init__()
        
        # Carrega o modelo ResNet50 com pesos pré-treinados na ImageNet.
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Congela os gradientes para todas as camadas do modelo.
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Descongela os gradientes para as camadas do último bloco convolucional (layer4).
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Substitui a camada de classificação final por uma nova, com o número correto de classes.
        num_ftrs = self.resnet.fc.in_features
        
        # Substitui a camada final por uma Sequência para incluir Dropout
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Adiciona Dropout (p=0.5) para regularização
            nn.Linear(num_ftrs, num_classes)
        )

    # Define o passo forward da rede neural.
    def forward(self, x):
        return self.resnet(x)


class MelTensorDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        
        # lê os nomes das pastas para descobrir as classes
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.filepaths = []
        self.labels = []
        
        for cls in self.classes:
            cls_folder = os.path.join(root, cls)
            for filename in os.listdir(cls_folder):
                if filename.endswith('.pt'):
                    self.filepaths.append(os.path.join(cls_folder, filename))
                    self.labels.append(self.class_to_idx[cls])
                    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        caminho = self.filepaths[idx]
        label = self.labels[idx]
        
        tensor = torch.load(caminho)
        
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, label



# BLOCO DE EXECUÇÃO PRINCIPAL
# Garante que o código de treinamento só seja executado quando o script for chamado diretamente.
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Define o caminho para a pasta principal do dataset.
    data_path = "./dataset_instrumentos_final"
    if not os.path.exists(data_path):
        print(f"ERRO: Pasta do dataset '{data_path}' não encontrada. Abortando.")
        exit()
        
    print(f"Usando dataset em: {data_path}")
    
    # Carrega os dados de treino
    train_dataset = MelTensorDataset(root=os.path.join(data_path, 'train'), transform=transform_treino)
    
    # Carrega os dados de teste (validação)
    # Assume que você tem uma pasta 'test' (ou 'val') no mesmo nível da 'train'
    test_path = os.path.join(data_path, 'test')
    if not os.path.exists(test_path):
        print(f"AVISO: Pasta de teste '{test_path}' não encontrada. A validação não será executada.")
        test_loader = None
        test_dataset_size = 0
    else:
        test_dataset = MelTensorDataset(root=test_path, transform=transform_teste)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_dataset_size = len(test_dataset)

    # Cria o DataLoader de treino
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_dataset_size = len(train_dataset)

    # Obtém o número de classes a partir do dataset
    num_classes = len(train_dataset.classes)
    print(f"Classes detectadas ({num_classes}): {train_dataset.classes}")
    print(f"Dispositivo de treinamento: {device}")

    # INICIALIZAÇÃO E TREINAMENTO
    
    # Instancia o modelo e move para o dispositivo (GPU/CPU)
    model = FineTunedResNet(num_classes=num_classes).to(device)

    # Define a função de perda (Loss Function)
    criterion = nn.CrossEntropyLoss()

    # Define o otimizador
    # Filtra parâmetros para otimizar apenas aqueles que não estão congelados (requires_grad=True)
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    
    optimizer = optim.AdamW(params_to_update, lr=0.001, weight_decay=1e-4)

    # Define o agendador de taxa de aprendizado (Learning Rate Scheduler)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    print(f"\nIniciando treinamento por {NUM_EPOCHS} épocas")
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_scores': [],
        }

    # Loop principal de treinamento
    for epoch in range(NUM_EPOCHS):
        # --- Fase de Treinamento ---
        model.train()  # Define o modelo para o modo de treinamento
        running_loss = 0.0
        running_correct = 0.0
        
        # TQDM para a barra de progresso do TREINO
        train_progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS} [Treino]", unit="batch")

        for inputs, labels in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Atualiza a descrição da barra de progresso
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / train_dataset_size
        epoch_train = 100 * running_correct / train_dataset_size

        # --- Fase de Validação ---
        if test_loader:
            model.eval()  # Define o modelo para o modo de avaliação
            val_loss = 0.0
            correct = 0

            with torch.no_grad():
                # TQDM para a barra de progresso da VALIDAÇÃO
                val_progress_bar = tqdm(test_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS} [Validação]", unit="batch")
                
                all_preds = []
                all_labels = []
                
                for inputs, labels in val_progress_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    
                    #guarda as predições e rótulos para cálculo de métricas adicionais
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # calcula métricas adicionais
            epoch_val_loss = val_loss / test_dataset_size
            epoch_acc = 100 * correct / test_dataset_size
            
            epoca_f1 = f1_score(all_labels, all_preds, average='weighted')
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_train)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_acc)
            history['val_f1_scores'].append(epoca_f1)
            
            print(f"\nFim da Época {epoch+1}: "
                  f"Loss Treino: {epoch_loss:.4f} | "
                  f"Acurácia Treino: {epoch_train:.2f}% | "
                  f"Loss Validação: {epoch_val_loss:.4f} | "
                  f"Acurácia Validação: {epoch_acc:.2f}% | "
                  f"F1 Score Validação: {epoca_f1:.4f} ")
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_acc,
                    'num_classes': num_classes,
                    'class_to_idx': train_dataset.class_to_idx
                }
                
                torch.save(checkpoint, MODEL_SAVE_PATH)
                print(f"\nCheckpoint salvo em {MODEL_SAVE_PATH} com Acurácia: {best_acc:.2f}%")
        else:
            # Caso não haja dados de teste
            print(f"\nFim da Época {epoch+1}: Loss Treino: {epoch_loss:.4f}")

        # Atualiza o agendador de taxa de aprendizado
        scheduler.step()

    print("\nTreinamento concluído.")

    # GERAÇÃO DE GRÁFICOS E MÉTRICAS FINAIS 
    # 1. Gráficos de Evolução (Loss e Accuracy)
    plt.figure(figsize=(14, 5))
    
    # Gráfico de Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Treino', marker='o')
    plt.plot(history['val_loss'], label='Validação', marker='o')
    plt.title('Evolução da Loss (Perda)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Gráfico de Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Treino', marker='o')
    plt.plot(history['val_acc'], label='Validação', marker='o')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('./graficos_evolucao_treinamento.png')
    plt.close()

    # 2. Matriz de Confusão do Melhor Modelo
    print("\nGerando Matriz de Confusão com o melhor modelo salvo...")
    if test_loader:
        # Carrega os pesos do melhor checkpoint
        model.load_state_dict(torch.load(MODEL_SAVE_PATH)['model_state_dict'])
        model.eval()
        
        final_preds = []
        final_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.numpy())

        # Calcula e plota a Matriz de Confusão
        cm = confusion_matrix(final_labels, final_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='d')
        plt.title('Matriz de Confusão - Melhor Modelo')
        plt.tight_layout()
        plt.savefig('./matriz_confusao.png')
        plt.close()
