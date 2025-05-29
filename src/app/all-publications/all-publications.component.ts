import {Component, OnInit} from '@angular/core';
import {Publication} from '../DTO/Publication';
import {Router} from '@angular/router';
import {UserService} from '../Service/UserService';
import {PublicationService} from '../Service/Publication.service';

@Component({
  selector: 'app-all-publications',
  imports: [],
  templateUrl: './all-publications.component.html',
  standalone: true,
  styleUrl: './all-publications.component.css'
})
export class AllPublicationsComponent implements OnInit{
  publications: Publication[] = [];
  currentUserRole: string = 'USER';
  loading: boolean = true;

  constructor(private publicationService: PublicationService, private userService: UserService, private router: Router) {
  }

  ngOnInit(): void {
    this.currentUserRole = localStorage.getItem('role') || "USER";
    this.loadPublications();
  }

  loadPublications(): void {
    this.publicationService.getAll().subscribe({
      next: (data) => {
        // Для USER показываем только published === true
        this.publications = data.filter(pub => pub.published || this.currentUserRole === 'ADMIN');
        this.loading = false;
      },
      error: (err) => {
        console.error('Ошибка загрузки публикаций', err);
        this.publications = [];
        this.loading = false;
      }
    });
  }

  goToPublication(id: number): void {
    this.router.navigate(['/publication', id]);
  }

}
