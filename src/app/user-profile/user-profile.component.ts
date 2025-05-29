import {Component, OnInit} from '@angular/core';
import {UserModel} from '../DTO/UserModel';
import {Publication} from '../DTO/Publication';
import {UserModelService} from '../Service/UserModel.service';
import {PublicationService} from '../Service/Publication.service';
import {ActivatedRoute, Router} from '@angular/router';
import {UserService} from '../Service/UserService';
import {environment} from '../../environments/environment';

@Component({
  selector: 'app-user-profile',
  imports: [],
  templateUrl: './user-profile.component.html',
  standalone: true,
  styleUrl: './user-profile.component.css'
})
export class UserProfileComponent implements OnInit{
  user!: any;
  models: UserModel[] = [];
  publications: Publication[] =[];
  lossImageSrc: string | null = null; // Для графика потерь
  accuracyImageSrc: string | null = null; // Для графика точности

  constructor(private userService: UserService, private modelService: UserModelService, private publicationService: PublicationService, private route: ActivatedRoute, private router: Router) {
  }

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      const userId = params.get('id');
      if(userId){
        this.userService.getById(userId).subscribe(user => {
          this.user = user
          this.modelService.getByUserId(this.user.id).subscribe(models =>this.models = models);
          this.publicationService.getPublicationByUserId(this.user.id).subscribe(publications =>this.publications = publications);
        });
      }
    });
  }

  createPublication(){
    localStorage.setItem('userId', this.user.id.toString());
    this.router.navigate([`/new-publication`]).then();
  }

  goToPublication(id: number){
    this.router.navigate([`/publication/${id}`]).then();
  }
  protected readonly environment = environment;
}
