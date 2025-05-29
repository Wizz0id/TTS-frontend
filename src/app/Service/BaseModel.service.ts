import {Injectable} from '@angular/core';
import {environment} from '../../environments/environment';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';
import {BaseModel} from '../DTO/BaseModel';

@Injectable({
  providedIn: "root"
})
export class BaseModelService{
  private modelUrl = `${environment.apiUrl}/${environment.baseModelUrl}`;

  constructor(private http: HttpClient) {
  }
  getModels(): Observable<BaseModel[]>{
    return this.http.get<BaseModel[]>(`${this.modelUrl}`)
  }
  addModel(model: BaseModel): Observable<BaseModel>{
    return this.http.post<BaseModel>(this.modelUrl, model);
  }
}
