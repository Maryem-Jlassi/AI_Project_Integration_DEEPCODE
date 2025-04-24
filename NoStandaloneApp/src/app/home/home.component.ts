import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError } from 'rxjs/operators';  // Import catchError from rxjs
import { of } from 'rxjs';  // Import 'of' from rxjs for handling errors

@Component({
  selector: 'app-home',
  standalone: false,
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {
  question: string = '';  // User input question
  answer: string = '';    // Answer from the API
  isLoading: boolean = false;  // Show loading status

  constructor(private http: HttpClient) {
    this.loadChatbot();  // Call to load the chatbot when the component is initialized
  }

  // Load the chatbot on page load
  loadChatbot(): void {
    this.http.get('http://localhost:8000/load').subscribe(
      (response: any) => {
        console.log(response.message);  // Log success message
      },
      error => {
        console.error('Error loading chatbot', error);
      }
    );
  }

  // Ask question to the FastAPI backend using async/await
  async askQuestion(): Promise<void> {
    if (!this.question.trim()) return;  // Prevent asking empty questions

    this.isLoading = true;  // Start loading

    try {
      const response: any = await this.http.post('http://localhost:8000/ask', { question: this.question })
        .pipe(
          catchError(err => {
            console.error('Error asking question', err);
            this.isLoading = false;  // Stop loading
            return of({ answer: 'Sorry, there was an error processing your request.' });  // Default error message
          })
        ).toPromise();  // Convert Observable to Promise using 'toPromise'
        
      this.answer = response.answer;  // Store the answer from the API
    } catch (error) {
      console.error('Error asking question', error);
      this.answer = 'An unexpected error occurred. Please try again.';  // Handle the case of unexpected errors
    } finally {
      this.isLoading = false;  // Stop loading
    }
  }

  // Dislike button function to send data to backend for saving to a file
  dislikeAnswer(): void {
    const dislikedData = { question: this.question, answer: this.answer };

    this.http.post('http://localhost:8000/dislike', dislikedData).subscribe(
      (response) => {
        console.log('Disliked answer saved successfully', response);
      },
      (error) => {
        console.error('Error saving disliked answer', error);
      }
    );
  }
}
