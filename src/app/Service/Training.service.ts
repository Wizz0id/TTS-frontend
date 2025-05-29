import {Injectable} from '@angular/core';
import {environment} from '../../environments/environment';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';
import {TrainingRequest} from '../DTO/TrainingRequest';
import {TrainingResult} from '../DTO/TrainingResult';

@Injectable({
  providedIn: 'root'
})
export class TrainingService{
  private trainUrl = `${environment.apiUrl}/${environment.userModelUrl}`;

  constructor(private http: HttpClient) {
  }

  generateToken(model: string): Observable<{ "token": string}> {
    return this.http.get<{ "token": string}>(`${this.trainUrl}/token?id=1&base=${model}`);
  }

  startTraining(request: TrainingRequest): Observable<TrainingResult> {
    return this.http.post<TrainingResult>(`${this.trainUrl}/start`, request);
  }
}
