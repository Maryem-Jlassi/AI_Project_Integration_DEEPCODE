import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  standalone: false,
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'NoStandaloneApp';

  isChatOpen = false;

  // 👇 Méthode pour ouvrir ou fermer la bulle de chat
  toggleChat() {
    this.isChatOpen = !this.isChatOpen;
  }
}
