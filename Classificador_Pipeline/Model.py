import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_WORKERS = 2
MODEL_SAVE_PATH = './model_checkpoint/resnet50_instrumentos.pth'

transform_treino = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    T.FrequencyMasking(freq_mask_param = 30),
    T.TimeMasking(time_mask_param = 50),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_teste = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in self.resnet.parameters():
            param.requires_grad = False

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    data_path = "./dataset_instrumentos"
    if not os.path.exists(data_path):
        print(f"ERRO: Pasta do dataset '{data_path}' não encontrada. Abortando.")
        exit()

    print(f"Usando dataset em: {data_path}")

    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform_treino)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    train_dataset_size = len(train_dataset)

    val_path = os.path.join(data_path, 'validation')
    if os.path.exists(val_path):
        val_dataset = datasets.ImageFolder(root=val_path, transform=transform_teste)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        val_dataset_size = len(val_dataset)
    else:
        val_loader = None
        val_dataset_size = 0
        
        
    test_path = os.path.join(data_path, 'test')
    if os.path.exists(test_path):
        test_dataset = datasets.ImageFolder(root=test_path, transform=transform_teste)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_dataset_size = len(test_dataset)
    else:
        test_loader = None
        test_dataset_size = 0

    num_classes = len(train_dataset.classes)
    print(f"Classes detectadas ({num_classes}): {train_dataset.classes}")
    print(f"Dispositivo de treinamento: {device}")

    targets = train_dataset.targets
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)


    #TREINAMENTO
    model = FineTunedResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(params_to_update, lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    print(f"\nIniciando treinamento por {NUM_EPOCHS} épocas...")
    best_acc = 0.0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1_scores': []
    }

    EPOCA_DESCONGELAMENTO = 5

    for epoch in range(NUM_EPOCHS):

        if epoch == EPOCA_DESCONGELAMENTO:
            print(f"\n[!] Época {epoch+1}: Descongelando layer3 e layer4 para Fine-Tuning profundo...")

            for name, param in model.resnet.named_parameters():
                if "layer4" in name or "layer3" in name or "layer2" in name:
                    param.requires_grad = True

            params_to_update = [param for param in model.parameters() if param.requires_grad]
            # LR alterado de 1e-5 para 1e-4
            optimizer = optim.AdamW(params_to_update, lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        model.train()
        running_loss = 0.0
        running_correct = 0.0

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
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / train_dataset_size
        epoch_train = 100 * running_correct / train_dataset_size


        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                val_progress_bar = tqdm(val_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS} [Validação]", unit="batch")

                for inputs, labels in val_progress_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_val_loss = val_loss / val_dataset_size
            epoch_acc = 100 * correct / val_dataset_size
            epoca_f1 = f1_score(all_labels, all_preds, average='weighted')

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_train)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_acc)
            history['val_f1_scores'].append(epoca_f1)

            print(f"\nFim da Época {epoch+1}: "
                  f"Loss Treino: {epoch_loss:.4f} | Acurácia Treino: {epoch_train:.2f}% | "
                  f"Loss Validação: {epoch_val_loss:.4f} | Acurácia Validação: {epoch_acc:.2f}% | "
                  f"F1 Score Validação: {epoca_f1:.4f}")

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
                print(f"[*] Checkpoint salvo em {MODEL_SAVE_PATH} com Acurácia: {best_acc:.2f}%")

            scheduler.step(epoch_acc)

        else:
            print(f"\nFim da Época {epoch+1}: Loss Treino: {epoch_loss:.4f}")
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_train)

    print("\nTreinamento concluído.")

    if test_loader:
        print("\nGerando gráficos e métricas de desempenho final...")

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Treino', marker='o')
        if val_loader: plt.plot(history['val_loss'], label='Validação', marker='o')
        plt.title('Evolução da Loss (Perda)')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Treino', marker='o')
        if val_loader: plt.plot(history['val_acc'], label='Validação', marker='o')
        plt.title('Evolução da Acurácia')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig('graficos_evolucao_treinamento.png')
        plt.show()
        plt.close()

        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        final_preds = []
        final_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(final_labels, final_preds)

        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='d')

        plt.title('Matriz de Confusão - Desempenho por Instrumento', fontsize=16, pad=20)
        plt.xlabel('Predição do Modelo', fontsize=12, labelpad=10)
        plt.ylabel('Rótulo Real (Gabarito)', fontsize=12, labelpad=10)

        plt.tight_layout()
        plt.savefig('matriz_confusao.png')
        plt.show()
        plt.close()

        acc_global = accuracy_score(final_labels, final_preds) * 100
        f1_global = f1_score(final_labels, final_preds, average='weighted')

        console = Console()

        print("\n" + "="*40)
        print(f"{'MÉTRICAS FINAIS DO MODELO':^40}")
        print("="*40)
        print(f"Acurácia de Treino (Final):  {history['train_acc'][-1]:.2f}%")
        if val_loader: print(f"Acurácia de Validação (Best): {checkpoint['best_val_acc']:.2f}%")
        print(f"Acurácia Global (Teste):     {acc_global:.2f}%")
        print(f"F1-Score (Weighted):         {f1_global:.4f}")
        print("="*40)

        report_dict = classification_report(final_labels, final_preds, target_names=train_dataset.classes, output_dict=True)

        tabela = Table(title="\nDesempenho Detalhado por Instrumento", show_header=True, header_style="bold magenta")
        tabela.add_column("Instrumento", style="dim", width=20)
        tabela.add_column("Precisão", justify="right")
        tabela.add_column("Recall", justify="right")
        tabela.add_column("F1-Score", justify="right")
        tabela.add_column("Suporte", justify="right")

        for instrumento, metrics in report_dict.items():
            if instrumento in train_dataset.classes:
                tabela.add_row(
                    instrumento.upper(),
                    f"{metrics['precision']:.2f}",
                    f"{metrics['recall']:.2f}",
                    f"{metrics['f1-score']:.2f}",
                    str(int(metrics['support']))
                )

        console.print(tabela)
        console.print(Panel(f"[dim]Gráficos salvos em:[/dim] [bold]Raiz do projeto[/bold]", border_style="white"))