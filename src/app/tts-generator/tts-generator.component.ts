import {Component, OnInit} from '@angular/core';
import {BaseModel} from '../DTO/BaseModel';
import {FormsModule, ReactiveFormsModule} from '@angular/forms';
import {BaseModelService} from '../Service/BaseModel.service';
import {environment} from '../../environments/environment';
import {SpeechService} from '../Service/Speech.service';
import {GeneratedSpeech} from '../DTO/GeneratedSpeech';
import {Voice} from '../DTO/Voice';

@Component({
  selector: 'app-tts-generator',
  imports: [
    ReactiveFormsModule,
    FormsModule
  ],
  templateUrl: './tts-generator.component.html',
  standalone: true,
  styleUrl: './tts-generator.component.css'
})
export class TtsGeneratorComponent implements OnInit {
  selectedModel: BaseModel | null = null;
  voice: Voice| null = null;
  baseModels: BaseModel[] = [];
  inputText: string = '';
  generatedAudioPath: string| null = null;

  constructor(private modelService: BaseModelService, private speechService: SpeechService) {
  }

  ngOnInit(): void {
    this.modelService.getModels().subscribe(models => {
      this.baseModels = models;
    })
  }

  generateVoice(){
    if(this.selectedModel){
      const speech: GeneratedSpeech = {
        id: 0,
        text: this.inputText,
        pathToAudio: ''
      }
      this.speechService.generateSpeech(this.selectedModel.id, speech).subscribe(speech => {
          this.generatedAudioPath = environment.url + "/" + speech.pathToAudio.replace('uploads\\', '')
        }
      );
    }
  }

}
