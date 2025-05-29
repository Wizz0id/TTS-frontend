export interface UserHistory{
  id:number;
  userId: number;
  username: string;
  dateTime: Date;
  speechId: number;
  text: string;
  pathToAudio: string;
}
