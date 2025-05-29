import {Injectable} from '@angular/core';
import {environment} from '../../environments/environment';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';
import {GeneratedSpeech} from '../DTO/GeneratedSpeech';

@Injectable({
  providedIn: "root"
})
export class SpeechService{
  private speechUrl = `${environment.apiUrl}/${environment.baseModelUrl}`;

  constructor(private http: HttpClient) {
  }
  getSpeeches(): Observable<GeneratedSpeech[]>{
    return this.http.get<GeneratedSpeech[]>(`${this.speechUrl}`)
  }

  generateSpeech(modelId: number, speech: GeneratedSpeech): Observable<GeneratedSpeech>{
    return this.http.post<GeneratedSpeech>(`${environment.apiUrl}/${environment.speechUrl}/${modelId}/generate`, speech);
  }
}
