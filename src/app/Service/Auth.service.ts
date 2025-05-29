import {Injectable} from '@angular/core';
import {environment} from '../../environments/environment';
import {Observable} from 'rxjs';
import {User} from '../DTO/User';
import {HttpClient} from '@angular/common/http';

@Injectable({
  providedIn: "root"
})
export class AuthService{
  authUrl: string = environment.apiUrl + environment.authUrl;

  constructor(private http: HttpClient) {
  }

  register(user: User): Observable<User>{
    return this.http.post<User>(this.authUrl, user);
  }
  login(user: User): Observable<User> {
    return this.http.get<User>(this.authUrl);
  }
}
