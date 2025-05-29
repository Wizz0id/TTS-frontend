import { Component } from '@angular/core';
import {FormsModule} from '@angular/forms';
import {AuthService} from '../Service/Auth.service';
import {Router} from '@angular/router';
import {User} from '../DTO/User';

@Component({
  selector: 'app-auth',
  imports: [
    FormsModule
  ],
  templateUrl: './auth.component.html',
  standalone: true,
  styleUrl: './auth.component.css'
})
export class AuthComponent {
  isLogin = true;

  loginData = {
    username: '',
    password: ''
  };

  registerData = {
    username: '',
    password: '',
    confirm: ''
  };

  constructor(private authService: AuthService, private router: Router) {}

  toggleForm(): void {
    this.isLogin = !this.isLogin;
  }

  onLogin(): void {
    const user: User = {id: 0, username: this.loginData.username, password:this.loginData.password, role: "USER"}
    this.authService.login(user).subscribe({
      next: (res) => {
        localStorage.setItem('role', res.role);
        this.router.navigate(['/profile']).then();
      },
      error: (err) => {
        alert('Ошибка входа: ' + err.message);
      }
    });
  }

  onRegister(): void {
    const user: User = {id: 0, username: this.registerData.username, password:this.registerData.password, role: "USER"}
    this.authService.register(user).subscribe({
      next: (res) => {
        alert('Успешная регистрация');
        this.isLogin = true;
      },
      error: (err) => {
        alert('Не удалось зарегистрироваться: ' + err.message);
      }
    });
  }
}
