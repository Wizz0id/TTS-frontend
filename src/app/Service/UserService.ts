import {Injectable} from '@angular/core';
import {environment} from '../../environments/environment';
import {Observable} from 'rxjs';
import {User} from '../DTO/User';
import {HttpClient} from '@angular/common/http';

@Injectable({
  providedIn: "root"
})
export class UserService{
  private userUrl = `${environment.apiUrl}/${environment.userUrl}`;

  constructor(private http: HttpClient) {}

  getById(id: string): Observable<User>{
    return this.http.get<User>(`${this.userUrl}/${id}`);
  }
}
