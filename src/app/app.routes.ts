import { Routes } from '@angular/router';
import {TtsGeneratorComponent} from './tts-generator/tts-generator.component';
import {CreateUserModelComponent} from './create-user-model/create-user-model.component';
import {UserProfileComponent} from './user-profile/user-profile.component';
import {CreatePublicationComponent} from './create-publication/create-publication.component';
import {PublicationComponent} from './publication/publication.component';
import {AllPublicationsComponent} from './all-publications/all-publications.component';
import {AuthComponent} from './auth/auth.component';

export const routes: Routes = [
  {path: 'generate-sound', component: TtsGeneratorComponent},
  {path: 'learn-model', component: CreateUserModelComponent},
  {path: 'user/:id', component: UserProfileComponent},
  {path: 'auth', component: AuthComponent},
  {path: 'new-publication', component: CreatePublicationComponent},
  {path: 'publication', component: AllPublicationsComponent},
  {path: 'publication/:id', component: PublicationComponent},
];
