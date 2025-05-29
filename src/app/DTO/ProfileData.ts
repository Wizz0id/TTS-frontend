import {User} from './User';
import {UserModel} from './UserModel';
import {Publication} from './Publication';

export interface ProfileData {
  user: User;
  models: UserModel[];
  publications: Publication[];
}
