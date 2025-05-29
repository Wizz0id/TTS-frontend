import {Component} from '@angular/core';
import {Publication} from '../DTO/Publication';
import {UserModel} from '../DTO/UserModel';
import {ActivatedRoute, Router} from '@angular/router';
import {PublicationService} from '../Service/Publication.service';
import {UserService} from '../Service/UserService';
import {environment} from '../../environments/environment';

@Component({
  selector: 'app-publication',
  imports: [],
  templateUrl: './publication.component.html',
  standalone: true,
  styleUrl: './publication.component.css'
})
export class PublicationComponent {
  publication: Publication | null = null;
  model: UserModel | null = null;
  currentUserRole: string = 'USER';
  isLoading: boolean = true;

  constructor(private route:ActivatedRoute, private router: Router, private publicationService: PublicationService, private userService: UserService) {
    this.loadPublication();
  }

  private loadPublication(): void {
    const id = this.route.snapshot.paramMap.get('id');
    if (!id) {
      this.router.navigate(['/not-found']).then();
      return;
    }
    this.publicationService.getPublicationById(+id).subscribe({
      next: (pub) => {
        this.publication = pub;
        this.model = pub.model;
        this.isLoading = false;
        this.currentUserRole = localStorage.getItem("role") || 'USER';
      },
      error: () => {
        this.router.navigate(['/not-found']).then();
      }
    });
  }

  publishPublication(): void {
    if (this.currentUserRole !== 'ADMIN') return;
    if(!this.publication) return;
    this.publicationService.publishPublication(this.publication.id).subscribe({
      next: (updatedPublication) => {
        this.publication = updatedPublication; // обновляем данные
        alert('Публикация успешно опубликована!');
      },
      error: (err) => {
        console.error('Ошибка при публикации:', err);
        alert('Не удалось опубликовать публикацию.');
      }
    });
  }
  deniedPublication(){

  }

  protected readonly environment = environment;
}
