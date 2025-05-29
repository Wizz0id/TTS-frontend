import {ChangeDetectorRef, Component, ElementRef, model, OnInit, ViewChild} from '@angular/core';
import {TrainingRequest} from '../DTO/TrainingRequest';
import {TrainingService} from '../Service/Training.service';
import {FormsModule} from '@angular/forms';
import {BaseModel} from '../DTO/BaseModel';
import {HttpClient} from '@angular/common/http';
import {BaseModelService} from '../Service/BaseModel.service';

@Component({
  selector: 'app-create-user-model',
  imports: [
    FormsModule,
  ],
  templateUrl: './create-user-model.component.html',
  standalone: true,
  styleUrl: './create-user-model.component.css'
})
export class CreateUserModelComponent implements OnInit{
  @ViewChild('notificationContainer') notificationContainer!: ElementRef;

  baseModels: BaseModel[] = [];
  trainingRequest: TrainingRequest = {
    token: '',
    trainingType: 'LOCAL',
    baseModel: this.baseModels[0],
    textPath: 'path/to/text',
    audioPath: 'path/to/audio'
  }

  constructor(private trainingService: TrainingService, private modelService: BaseModelService, private http: HttpClient) {}

  generateToken(): void {
    this.trainingService.generateToken(this.trainingRequest.baseModel.name).subscribe(res => {
      this.trainingRequest.token = res.token;
    });
  }

  generateColabLink(): string {
    return `https://colab.research.google.com/drive/1Pctygrf1i1AuvxHurpCBF1EX5CWKSOdm`;
  }

  downloadScript(): void {
    const token = this.trainingRequest.token;
    const baseModel = this.trainingRequest.baseModel;

    if (!token) {
      alert('Сначала сгенерируйте токен');
      return;
    }

    // Путь к файлу в assets
    this.http.get('/script/train_tts.py', { responseType: 'text' }).subscribe(data => {
      // Создаем Blob и скачиваем
      const blob = new Blob([data], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = 'train.py';
      a.click();

      URL.revokeObjectURL(url);
    });
  }

  protected readonly model = model;

  ngOnInit(): void {
    this.modelService.getModels().subscribe(models => {
      this.baseModels = models;
      this.trainingRequest.baseModel = models[0];
    })
  }

  copyToken(inputElement: HTMLDivElement): void {
    if (!this.trainingRequest.token) {
      alert('Нет токена для копирования');
      return;
    }

    navigator.clipboard.writeText(this.trainingRequest.token)
      .then(() => {
        // Показать уведомление
        this.showNotification('Токен скопирован');
      })
      .catch(error => {
        console.error('Ошибка при копировании:', error);
        alert('Не удалось скопировать токен');
      });
  }

  showNotification(message: string): void {
    const notification = document.createElement('div');
    notification.classList.add('notification');
    notification.textContent = message;

    // Добавляем уведомление в контейнер
    this.notificationContainer.nativeElement.appendChild(notification);

    // Установить начальное состояние (полная прозрачность)
    notification.style.opacity = '0';

    // Постепенно показываем уведомление
    setTimeout(() => {
      notification.style.opacity = '1';
    }, 100);

    // После 3 секунд начинаем скрывать уведомление
    setTimeout(() => {
      notification.style.opacity = '0';
      setTimeout(() => {
        this.notificationContainer.nativeElement.removeChild(notification);
      }, 300); // Задержка для анимации исчезновения
    }, 3000);
  }
}
