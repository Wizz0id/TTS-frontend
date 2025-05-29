import {Voice} from './Voice';

export interface BaseModel{
  id: number;
  name: string;
  voiceList: Voice[];
}
