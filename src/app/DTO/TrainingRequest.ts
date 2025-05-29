import {BaseModel} from './BaseModel';

export interface TrainingRequest{
  token:string;
  trainingType: 'LOCAL' | 'CLOUD';
  baseModel: BaseModel;
  textPath?: string;
  audioPath?: string;
}
