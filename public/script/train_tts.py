import subprocess
import sys

# ========== АВТОУСТАНОВКА ==========
required_packages = ["torch", "librosa", "matplotlib", "tensorboard", "numpy", "soundfile", "requests", "pillow"]
def install_dependencies():
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Установка {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install_dependencies()

# ========== ИМПОРТ ==========
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import librosa
import soundfile as sf
import requests
from io import BytesIO
from PIL import Image

# ========== Tacotron 2 MODEL ==========
class Tacotron2(nn.Module):
  def __init__(self, vocab_size=42, embedding_dim=256, mel_dim=80, hidden_dim=512):
    super(Tacotron2, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.decoder = nn.LSTM(mel_dim, hidden_dim * 2, batch_first=True)
    self.linear = nn.Linear(hidden_dim * 2, mel_dim)

  def forward(self, text_inputs, mel_inputs):
    embedded = self.embedding(text_inputs)
    encoder_outputs, _ = self.encoder(embedded)

    # Повторение скрытого состояния на длину mel
    decoder_outputs, _ = self.decoder(mel_inputs.transpose(1, 2))
    mel_outputs = self.linear(decoder_outputs)
    mel_outputs = mel_outputs.transpose(1, 2)  # [B, mel, T]

    return mel_outputs

# ============================
#        DATASET
# ============================
class TTSDataset(Dataset):
    def __init__(self, metadata_path, wav_dir, sample_rate=22050):
        with open(metadata_path, encoding='utf-8') as f:
            self.metadata = [line.strip().split("|") for line in f]
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # text, file_id = self.metadata[idx]
        file_id, text  = self.metadata[idx]
        wav_path = os.path.join(self.wav_dir, f"{file_id}.wav")

        wav, _ = librosa.load(wav_path, sr=self.sample_rate)
        text_seq = self.text_to_sequence(text)
        mel_spec = self.get_mel_spectrogram(wav)

        return torch.tensor(text_seq), torch.tensor(mel_spec)

    def text_to_sequence(self, text):
        # symbols = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя+."
        symbols = " qwertyuiopasdfghjklzxcvbnm+."
        symbol_to_id = {s: i+1 for i, s in enumerate(symbols)}
        return [symbol_to_id.get(c, 0) for c in text.lower()]

    def get_mel_spectrogram(self, wav):
        mel_spec = librosa.feature.melspectrogram(
            y=wav, sr=self.sample_rate, n_fft=1024, hop_length=256, n_mels=80)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec

# ============================
#       TRAINING LOOP
# ============================
def train(model, dataloader, optimizer, criterion, epochs, device, log_dir="runs", ckpt_dir="checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    model.train()

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        total_loss = 0
        for i, (text, mel) in enumerate(dataloader):
            text, mel = text.to(device), mel.to(device)
            mel_input = mel[:, :, :-1]
            mel_target = mel[:, :, 1:]

            mel_output = model(text, mel_input)
            loss = criterion(mel_output, mel_target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Step {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
                writer.add_scalar("Loss/train_step", loss.item(), epoch * len(dataloader) + i)

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        acc = 1 / (avg_loss + 1e-6)  # псевдо-точность
        acc_history.append(acc)

        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", acc, epoch)
        print(f"Epoch {epoch+1} завершена. Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        torch.save(model.state_dict(), f"{ckpt_dir}/epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), f"{ckpt_dir}/final_model.pt")
    writer.close()
    plot_metrics(loss_history, acc_history)

    # ========== ИНФЕРЕНС ПЕРВОЙ ФРАЗЫ ==========
    print("\nСинтез тестовой фразы (первая из датасета)...")
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(dataloader))
        text_input = first_batch[0][0].unsqueeze(0).to(device)  # [1, T]
        mel_input = torch.zeros(1, 80, 10).to(device)  # стартовая последовательность

        generated = []
        for _ in range(400):
            mel_output = model(text_input, mel_input)
            next_frame = mel_output[:, :, -1:].detach()
            generated.append(next_frame.squeeze(2).cpu().numpy()[0])  # [80]
            mel_input = torch.cat((mel_input, next_frame), dim=2)

        mel_result = np.stack(generated, axis=1)  # [80, T]
        plt.figure(figsize=(12, 4))
        plt.imshow(mel_result, aspect="auto", origin="lower", interpolation="none")
        plt.title("Mel-spectrogram Output (First Example)")
        plt.xlabel("Time")
        plt.ylabel("Mel Filterbank")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("test_mel_output.png")
        plt.close()
        print("Сохранено изображение: test_mel_output.png")

        # ========== ИНВЕРСИЯ MEL → WAV ==========
        print("Инвертирование мел-спектра в WAV...")

        # Обратное преобразование
        audio = librosa.feature.inverse.mel_to_audio(
            mel_result, sr=22050, n_fft=1024, hop_length=256, win_length=1024, power=2.0, n_iter=60)

        # Сохранение
        sf.write("test_output.wav", audio, samplerate=22050)
        print("Сохранено аудио: test_output.wav")

    # ========== СЧИТЫВАНИЕ МЕТРИК ==========
    final_loss = loss_history[-1] if loss_history else 0
    avg_last_3 = np.mean(loss_history[-3:]) if len(loss_history) >= 3 else final_loss
    pseudo_acc = acc_history[-1] if acc_history else 0

    metrics = {
      'loss_final': final_loss,
      'loss_avg_last_3': avg_last_3,
      'pseudo_accuracy': pseudo_acc
    }

    return metrics

# ============================
#     ГРАФИКИ МЕТРИК
# ============================
def plot_metrics(losses, accuracies):
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.savefig("loss_plot.png")
    plt.close()

    plt.figure()
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (1 / loss)')
    plt.title('Pseudo Accuracy')
    plt.grid()
    plt.savefig("accuracy_plot.png")
    plt.close()

# ============================
#       ПАДДИНГ ДАННЫХ
# ============================
def collate_fn(batch):
    texts, mels = zip(*batch)
    max_len_text = max([len(t) for t in texts])
    max_len_mel = max([m.shape[1] for m in mels])

    padded_texts = torch.zeros(len(batch), max_len_text, dtype=torch.long)
    padded_mels = torch.zeros(len(batch), 80, max_len_mel)

    for i in range(len(batch)):
        padded_texts[i, :len(texts[i])] = texts[i]
        padded_mels[i, :, :mels[i].shape[1]] = mels[i]

    return padded_texts, padded_mels

# ============================
#       SEND TO SERVER
# ============================
def send_to_server(token, model_path, loss_plot_path, accuracy_plot_path, metrics):
    url = "http://localhost:8080/api/user_models/new_model"

    files = {
        'model': open(model_path, 'rb'),
        'lossPlot': open(loss_plot_path, 'rb'),
        'accuracyPlot': open(accuracy_plot_path, 'rb'),
    }

    data = {
        'token': token,
        'metric1': str(metrics['loss_final']),
        'metric2': str(metrics['loss_avg_last_3']),
        'metric3': str(metrics['pseudo_accuracy'])
    }

    try:
        response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            print("✅ Успешно отправлено на сервер.")
        else:
            print("❌ Ошибка при отправке:", response.text)
    except Exception as e:
        print("⚠️ Не удалось подключиться к серверу:", e)

# ============================
#            MAIN
# ============================
def main():
    parser = argparse.ArgumentParser(description="Train Tacotron 2 on Russian dataset")
    parser.add_argument("--token", type=str, required=True, help="User auth token")
    parser.add_argument("--metadata", type=str, required=True, help="Path to .txt file with lines 'text|file_id'")
    parser.add_argument("--wav_dir", type=str, required=True, help="Path to folder with WAV files")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TTSDataset(metadata_path=args.metadata, wav_dir=args.wav_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = Tacotron2().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Начало обучения...")
    metrics = train(model, dataloader, optimizer, criterion, epochs=args.epochs, device=device)
    print("Обучение завершено")
    send_to_server(
      token=args.token,
      model_path="checkpoints/final_model.pt",
      loss_plot_path="loss_plot.png",
      accuracy_plot_path="accuracy_plot.png",
      metrics=metrics
    )

if __name__ == "__main__":
    main()
