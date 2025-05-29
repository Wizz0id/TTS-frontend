import {Component} from '@angular/core';
import {UserService} from '../Service/UserService';
import {UserModelService} from '../Service/UserModel.service';
import {User} from '../DTO/User';
import {UserModel} from '../DTO/UserModel';
import {Publication} from '../DTO/Publication';
import {FormsModule} from '@angular/forms';
import {PublicationService} from '../Service/Publication.service';

@Component({
  selector: 'app-create-publication',
  imports: [
    FormsModule
  ],
  templateUrl: './create-publication.component.html',
  standalone: true,
  styleUrl: './create-publication.component.css'
})
export class CreatePublicationComponent{
  user: User| null = null;
  userModels: UserModel[] = [];
  publication: Publication = {
    id: 0,
    title: '',
    description: '',
    username: '',
    published: false,
    model: {} as UserModel
  };

  constructor(private userService: UserService, private  modelService: UserModelService, private publicationService: PublicationService) {
    const id = localStorage.getItem("userId");
    if(id){
      userService.getById(id).subscribe(user => {
        this.user = user
        this.publication.username = user.username;
      })
      modelService.getByUserId(id).subscribe(models => this.userModels = models)
    }
  }
  onSubmit(){
    if(this.user)
      this.publicationService.createPublication(this.publication, this.user.id).subscribe({
        next: (response) => {
          alert(response.message);
        },
        error: (error)=>{
          alert(`Произошла ошибка при отправке публикации:${error}`);
        }
      });
  }
}
