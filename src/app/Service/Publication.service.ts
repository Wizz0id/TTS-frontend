import {Injectable} from '@angular/core';
import {environment} from '../../environments/environment';
import {Observable} from 'rxjs';
import {Publication} from '../DTO/Publication';
import {HttpClient} from '@angular/common/http';

@Injectable({
  providedIn: "root"
})
export class PublicationService{
  private modelUrl = `${environment.apiUrl}/${environment.publicationUrl}`;

  constructor(private http: HttpClient) {
  }

  getAll():Observable<Publication[]>{
    return this.http.get<Publication[]>(`${this.modelUrl}`);
  }

  getPublicationById(id: number): Observable<Publication>{
    return this.http.get<Publication>(`${this.modelUrl}/${id}`);
  }

  getPublicationByUserId(userId: number): Observable<Publication[]>{
    return this.http.get<Publication[]>(`${this.modelUrl}?userId=${userId}`);
  }

  createPublication(publication: Publication, userId: number): Observable<{ message: string }>{
    return this.http.post<{ message: string }>(`${this.modelUrl}?userId=${userId}`, publication);
  }

  publishPublication(id: number): Observable<Publication>{
    return this.http.get<Publication>(`${this.modelUrl}/${id}/publish`);
  }
}
