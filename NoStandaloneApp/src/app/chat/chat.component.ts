import { Component, OnInit, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormControl } from '@angular/forms';
import { ReactiveFormsModule } from '@angular/forms';
import { CommonModule, DatePipe } from '@angular/common';

interface ChatMessage {
  content: string;
  isUser: boolean;
  sources?: string[];
  timestamp: Date;
}

interface RagResponse {
  answer: string;
  sources: string[];
}

@Component({
    selector: 'app-chat',
    templateUrl: './chat.component.html',
    styleUrls: ['./chat.component.css'],
    standalone: true,
    imports: [ReactiveFormsModule, CommonModule, DatePipe]
  })
export class ChatComponent implements OnInit {
  @ViewChild('chatContainer') chatContainer!: ElementRef;
  @ViewChild('audioFileInput') audioFileInput!: ElementRef;
  
  messages: ChatMessage[] = [];
  questionInput = new FormControl('');
  isLoading = false;
  isRecording = false;
  showSources = false;
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private readonly API_URL = 'http://localhost:8000';
  
  constructor(private http: HttpClient) { }

  ngOnInit(): void {
    this.addBotMessage('Bonjour, je suis le chatbot RH d\'ACTIA. Comment puis-je vous aider ?');
  }

  sendQuestion(): void {
    const question = this.questionInput.value?.trim();
    if (!question) return;
  
    this.addUserMessage(question);
    this.questionInput.setValue('');
    this.isLoading = true;
  
    // Ajoute un message temporaire de bot
    const tempBotMessage: ChatMessage = {
      content: '✍️ Rédaction en cours...',
      isUser: false,
      timestamp: new Date(),
      sources: []
    };
    this.messages.push(tempBotMessage);
    this.scrollToBottom();
  
    this.http.post<RagResponse>(`${this.API_URL}/query`, { query: question }).subscribe({
      next: (response) => {
        // Remplace le message temporaire par la réponse réelle
        this.messages[this.messages.indexOf(tempBotMessage)] = {
          content: response.answer,
          isUser: false,
          sources: response.sources,
          timestamp: new Date()
        };
        this.isLoading = false;
        this.scrollToBottom();
      },
      error: (error) => {
        console.error('Erreur lors de l\'envoi de la question', error);
        this.messages[this.messages.indexOf(tempBotMessage)] = {
          content: 'Désolé, une erreur est survenue. Veuillez réessayer.',
          isUser: false,
          timestamp: new Date()
        };
        this.isLoading = false;
      }
    });
  }
  

  startRecording(): void {
    // Si déjà en train d'enregistrer, ne rien faire
    if (this.isRecording) {
        return;
    }

    this.isRecording = true;
    this.audioChunks = [];
    
    navigator.mediaDevices.getUserMedia({ 
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 16000,
            channelCount: 1
        }
    })
    .then(stream => {
        this.mediaStream = stream;
        
        this.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.audioChunks.push(event.data);
            }
        };
        
        this.mediaRecorder.onstop = () => {
            const audioBlob = new Blob(this.audioChunks, { 
                type: 'audio/webm;codecs=opus' 
            });
            
            // Envoyer l'audio
            this.sendAudioQuestion(audioBlob);
            
            // Ne pas nettoyer ici, cela sera fait dans stopRecording
        };
        
        this.mediaRecorder.start();
    })
    .catch(error => {
        console.error('Erreur microphone:', error);
        this.isRecording = false;
        this.addBotMessage('Microphone inaccessible. Veuillez vérifier les permissions.');
        // Nettoyer en cas d'erreur
        this.cleanupMedia();
    });
}

stopRecording(): void {
    if (this.mediaRecorder && this.isRecording) {
        this.isRecording = false;
        
        try {
            // Seulement arrêter l'enregistrement si l'état est "recording"
            if (this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
            }
        } catch (error) {
            console.error('Erreur lors de l\'arrêt de l\'enregistrement:', error);
        } finally {
            // Toujours nettoyer les ressources après avoir arrêté l'enregistrement
            this.cleanupMedia();
        }
    }
}

// Méthode pour nettoyer les ressources
private cleanupMedia(): void {
    // Arrêter les pistes audio
    if (this.mediaStream) {
        this.mediaStream.getTracks().forEach(track => track.stop());
        this.mediaStream = null;
    }
    
    // Réinitialiser le recorder
    this.mediaRecorder = null;
}

sendAudioQuestion(audioBlob: Blob): void {
  const tempBotMessage: ChatMessage = {
    content: '🧠 Analyse de votre message audio...',
    isUser: false,
    timestamp: new Date(),
    sources: []
  };

  this.messages.push({
    content: '🎤 Question audio...',
    isUser: true,
    timestamp: new Date()
  });

  this.messages.push(tempBotMessage);
  this.isLoading = true;
  this.scrollToBottom();

  const formData = new FormData();
  const audioFile = new File([audioBlob], 'audio_message.webm', { type: 'audio/webm;codecs=opus' });
  formData.append('file', audioFile);

  this.http.post<RagResponse>(`${this.API_URL}/query/audio`, formData).subscribe({
    next: (response) => {
      this.messages[this.messages.indexOf(tempBotMessage)] = {
        content: response.answer,
        isUser: false,
        sources: response.sources,
        timestamp: new Date()
      };
      this.isLoading = false;
      this.scrollToBottom();
    },
    error: (error) => {
      console.error('Erreur API audio:', error);
      this.messages[this.messages.indexOf(tempBotMessage)] = {
        content: 'Erreur de traitement audio. Veuillez réessayer.',
        isUser: false,
        timestamp: new Date()
      };
      this.isLoading = false;
    }
  });
}

  uploadFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    
    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    this.isLoading = true;
    this.addBotMessage(`Téléchargement et indexation du document "${file.name}" en cours...`);
    
    this.http.post(`${this.API_URL}/upload/document`, formData).subscribe({
      next: (response: any) => {
        this.addBotMessage(`✅ ${response.message}`);
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Erreur lors du téléchargement du document', error);
        this.addBotMessage('❌ Erreur lors du téléchargement du document.');
        this.isLoading = false;
      }
    });
    
    // Réinitialiser l'input file
    this.audioFileInput.nativeElement.value = '';
  }

  toggleSources(): void {
    this.showSources = !this.showSources;
  }

  private addUserMessage(content: string): void {
    this.messages.push({
      content,
      isUser: true,
      timestamp: new Date()
    });
    this.scrollToBottom();
  }

  private addBotMessage(content: string, sources: string[] = []): void {
    this.messages.push({
      content,
      isUser: false,
      sources,
      timestamp: new Date()
    });
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      if (this.chatContainer) {
        const element = this.chatContainer.nativeElement;
        element.scrollTop = element.scrollHeight;
      }
    }, 100);
  }
  
  hasMessagesWithSources(): boolean {
    return this.messages.some(m => m.sources && m.sources.length > 0);
  }
}