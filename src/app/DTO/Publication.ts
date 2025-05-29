import {UserModel} from './UserModel';

export interface Publication{
  id:number;
  title:string;
  description:string;
  username: string;
  published: false;
  model: UserModel;
}
