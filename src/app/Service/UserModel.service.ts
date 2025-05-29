import {Injectable} from '@angular/core';
import {environment} from '../../environments/environment';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';
import {UserModel} from '../DTO/UserModel';

@Injectable({
  providedIn: "root"
})
export class UserModelService {
  private modelUrl = `${environment.apiUrl}/${environment.userModelUrl}`;

  constructor(private http: HttpClient) {}

  getByUserId(userId: string): Observable<UserModel[]>{
    return this.http.get<UserModel[]>(`${this.modelUrl}?userId=${userId}`);
  }
}
