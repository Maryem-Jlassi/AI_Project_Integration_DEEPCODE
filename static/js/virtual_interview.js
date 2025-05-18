// Global state
let localStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isVideoEnabled = false;
let isAudioEnabled = false;
let isInterviewerSpeaking = false;
let candidateName = 'Candidate';
let currentPosition = 'Software Developer';
let isRecording = false;
let isInterviewActive = false;
let currentInterviewId = null;

// Speech recognition
let recognition = null;
let recognitionActive = false;
let transcription = '';

// DOM elements
const interviewScreen = document.getElementById('interview-screen');
const interviewFinishedPage = document.getElementById('interview-finished-page');
const localVideo = document.getElementById('local-video');
const interviewerVideo = document.getElementById('interviewer-video');
const interviewerImage = document.getElementById('interviewer-image');
const toggleVideoBtn = document.getElementById('toggle-video');
const toggleAudioBtn = document.getElementById('toggle-audio');
const chatMessages = document.getElementById('chat-messages');
const textInput = document.getElementById('text-input');
const sendTextBtn = document.getElementById('send-text');
const candidateAudioIndicator = document.getElementById('candidateAudioIndicator');
const interviewerAudioIndicator = document.getElementById('interviewerAudioIndicator');
const jobTitleHeader = document.getElementById('jobTitleHeader');
const endCallBtnHeader = document.getElementById('endCallBtnHeader');
const endBtn = document.getElementById('endBtn');
const startNewInterviewBtn = document.getElementById('startNewInterviewBtn');
const headerCandidateNameSpan = document.getElementById('headerCandidateName');
const videoOverlayCandidateNameSpan = document.getElementById('videoOverlayCandidateName');

// Audio element for interviewer's speech
const interviewerAudio = document.getElementById('interviewer-audio');

// Initialize interviewer media
function initializeInterviewerMedia() {
    if (!interviewerVideo || !interviewerImage) {
        console.error('Interviewer media elements not found');
        addMessage('System', 'Error: Interviewer media elements not found.', 'system');
        return;
    }
    interviewerVideo.src = '/static/images/testav.mp4';
    interviewerVideo.loop = true;
    interviewerVideo.muted = true;
    interviewerVideo.style.display = 'none';
    interviewerImage.src = '/static/images/photo.jpg';
    interviewerImage.style.display = 'block';
    interviewerVideo.onerror = () => {
        console.error('Failed to load interviewer video');
        addMessage('System', 'Unable to load interviewer video.', 'system');
        interviewerVideo.style.display = 'none';
        interviewerImage.style.display = 'block';
    };
}

// Synchronize interviewer video/image with audio
function syncInterviewerMedia(audioUrl) {
    if (!interviewerVideo || !interviewerImage || !audioUrl) {
        console.warn('Missing interviewer media or audio URL');
        addMessage('System', 'Interviewer media or audio not available.', 'system');
        return;
    }

    // Make sure the video is ready
    if (interviewerVideo.readyState === 0) {
        interviewerVideo.src = '/static/images/testav.mp4';
        interviewerVideo.loop = true;
        interviewerVideo.muted = true;
        interviewerVideo.load();
    }

    console.log('Playing interviewer audio from URL:', audioUrl);
    
    // Check if the audio file exists before trying to play it
    fetch(audioUrl, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                console.log('Audio file exists, proceeding with playback');
                playInterviewerAudio(audioUrl);
            } else {
                console.error('Audio file not found:', audioUrl);
                addMessage('System', 'Interviewer audio file not found. Please try again.', 'system');
            }
        })
        .catch(error => {
            console.error('Error checking audio file:', error);
            addMessage('System', 'Error checking audio file. Please try again.', 'system');
        });
}

// Play interviewer audio and synchronize with video
function playInterviewerAudio(audioUrl) {
    interviewerAudio.src = audioUrl;
    
    // When audio starts playing, show video
    interviewerAudio.onplay = () => {
        console.log('Interviewer audio started playing, showing video');
        isInterviewerSpeaking = true;
        interviewerVideo.style.display = 'block';
        interviewerImage.style.display = 'none';
        
        // Make sure video plays
        interviewerVideo.currentTime = 0;
        interviewerVideo.play().catch(e => {
            console.error('Interviewer video playback error:', e);
            addMessage('System', 'Error playing interviewer video.', 'system');
            interviewerVideo.style.display = 'none';
            interviewerImage.style.display = 'block';
        });
        updateInterviewerAudioIndicator(true);
        
        // Stop speech recognition and recording when interviewer is speaking
        stopSpeechRecognition();
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            isRecording = false;
            candidateAudioIndicator.classList.remove('active');
        }
    };
    
    // When audio ends, hide video and show image
    interviewerAudio.onended = () => {
        console.log('Interviewer audio ended, hiding video');
        isInterviewerSpeaking = false;
        interviewerVideo.pause();
        interviewerVideo.style.display = 'none';
        interviewerImage.style.display = 'block';
        updateInterviewerAudioIndicator(false);
        
        // Clear any previous transcription
        transcription = '';
        
        // Make sure microphone is enabled
        if (localStream && localStream.getAudioTracks().length > 0) {
            localStream.getAudioTracks().forEach(track => {
                track.enabled = true;
            });
            isAudioEnabled = true;
            updateAudioButtonState();
        }
        
        // Show visual indicator that it's user's turn
        addMessage('System', 'Your turn to speak now...', 'system');
        
        // Start recording user's response after a short delay
        setTimeout(() => {
            if (!isInterviewerSpeaking) {
                try {
                    console.log('Starting to record user response');
                    
                    // Activate the candidate audio indicator
                    candidateAudioIndicator.classList.add('active');
                    candidateAudioIndicator.classList.add('pulse');
                    
                    // Start recording
                    if (mediaRecorder && mediaRecorder.state !== 'recording') {
                        mediaRecorder.start();
                        isRecording = true;
                    }
                    
                    // Start speech recognition with a slight delay to avoid residual audio
                    setTimeout(() => {
                        if (!isInterviewerSpeaking) {
                            startSpeechRecognition();
                            console.log('Speech recognition started automatically');
                        }
                    }, 300);
                    
                } catch (e) {
                    console.error('Failed to start recording after interviewer finished:', e);
                    addMessage('System', 'Error starting microphone. Please refresh the page.', 'system');
                }
            }
        }, 1000);
    };
    
    interviewerAudio.onerror = (e) => {
        console.error('Interviewer audio load error:', e);
        addMessage('System', 'Unable to load interviewer audio.', 'system');
        isInterviewerSpeaking = false;
        interviewerVideo.pause();
        interviewerVideo.style.display = 'none';
        interviewerImage.style.display = 'block';
        updateInterviewerAudioIndicator(false);
    };
    
    // Play the audio
    interviewerAudio.play().catch(e => {
        console.error('Interviewer audio playback error:', e);
        addMessage('System', 'Error playing interviewer audio.', 'system');
        interviewerVideo.style.display = 'none';
        interviewerImage.style.display = 'block';
    });
}

// Update candidate video button state
function updateVideoButtonState() {
    if (!toggleVideoBtn) return;
    const icon = toggleVideoBtn.querySelector('i');
    if (!localStream || localStream.getVideoTracks().length === 0) {
        icon.className = 'bi bi-camera-video-off-fill';
        toggleVideoBtn.classList.remove('active');
        toggleVideoBtn.disabled = true;
        return;
    }
    toggleVideoBtn.disabled = false;
    isVideoEnabled = localStream.getVideoTracks()[0].enabled;
    icon.className = isVideoEnabled ? 'bi bi-camera-video-fill' : 'bi bi-camera-video-off-fill';
    toggleVideoBtn.classList.toggle('active', isVideoEnabled);
    
    // Make sure the video element is visible if video is enabled
    if (localVideo) {
        if (isVideoEnabled) {
            localVideo.style.display = 'block';
        } else {
            localVideo.style.display = 'none';
        }
    }
}

// Update candidate audio button and indicator state
function updateAudioButtonState() {
    if (!toggleAudioBtn || !candidateAudioIndicator) return;
    const icon = toggleAudioBtn.querySelector('i');
    const candidateAudioIcon = candidateAudioIndicator.querySelector('i');
    if (!localStream || localStream.getAudioTracks().length === 0) {
        icon.className = 'bi bi-mic-mute-fill';
        toggleAudioBtn.classList.remove('active');
        toggleAudioBtn.classList.add('muted');
        toggleAudioBtn.disabled = true;
        candidateAudioIndicator.classList.remove('active');
        candidateAudioIndicator.classList.add('muted');
        candidateAudioIcon.className = 'bi bi-mic-mute-fill';
        candidateAudioIndicator.setAttribute('aria-label', 'Your audio muted');
        return;
    }
    toggleAudioBtn.disabled = false;
    isAudioEnabled = localStream.getAudioTracks()[0].enabled;
    icon.className = isAudioEnabled ? 'bi bi-mic-fill' : 'bi bi-mic-mute-fill';
    toggleAudioBtn.classList.toggle('active', isAudioEnabled);
    toggleAudioBtn.classList.toggle('muted', !isAudioEnabled);
    candidateAudioIndicator.classList.toggle('active', isAudioEnabled);
    candidateAudioIndicator.classList.toggle('muted', !isAudioEnabled);
    candidateAudioIcon.className = isAudioEnabled ? 'bi bi-mic-fill' : 'bi bi-mic-mute-fill';
    candidateAudioIndicator.setAttribute('aria-label', isAudioEnabled ? 'Your audio enabled' : 'Your audio muted');
}

// Update interviewer audio indicator
function updateInterviewerAudioIndicator(isPlaying) {
    if (!interviewerAudioIndicator) return;
    const icon = interviewerAudioIndicator.querySelector('i');
    interviewerAudioIndicator.classList.toggle('active', isPlaying);
    interviewerAudioIndicator.classList.toggle('muted', !isPlaying);
    interviewerAudioIndicator.classList.toggle('pulse', isPlaying);
    icon.className = isPlaying ? 'bi bi-volume-up-fill' : 'bi bi-volume-mute-fill';
    interviewerAudioIndicator.setAttribute('aria-label', isPlaying ? 'Interviewer speaking' : 'Interviewer silent');
}

// Add CSS for pulse animation
function addPulseAnimation() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
        .audio-indicator.pulse {
            animation: pulse 1.5s infinite;
        }
        .audio-indicator.active.pulse {
            background-color: rgba(92, 184, 92, 0.7);
        }
    `;
    document.head.appendChild(style);
}

// Initialize media devices (camera and microphone)
function initializeMediaDevices() {
    console.log('Requesting media permissions...');
    
    // Request persistent permissions for microphone and camera
    navigator.mediaDevices.getUserMedia({ 
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
        }, 
        video: true 
    })
    .then(stream => {
        console.log('Media permissions granted');
        localStream = stream;
        
        if (localVideo) {
            localVideo.srcObject = stream;
            localVideo.onloadedmetadata = () => {
                localVideo.play().catch(e => console.error('Error playing local video:', e));
            };
        }
        
        // Ensure audio and video are enabled by default
        if (stream.getAudioTracks().length > 0) {
            stream.getAudioTracks().forEach(track => {
                track.enabled = true;
            });
            isAudioEnabled = true;
        }
        
        if (stream.getVideoTracks().length > 0) {
            stream.getVideoTracks().forEach(track => {
                track.enabled = true;
            });
            isVideoEnabled = true;
        }
        
        updateVideoButtonState();
        updateAudioButtonState();
        
        // Set up audio recording
        setupAudioRecording();
        
        // Set up speech recognition
        setupSpeechRecognition();
        
        // Add welcome message
        addMessage('System', 'Welcome to the virtual interview. The interviewer will begin shortly...', 'system');
        
        // Start the interview
        startInterview();
    })
        .catch(error => {
            console.error('Error accessing media devices:', error);
            
            // Try with just audio if video fails
            if (error.name === 'NotAllowedError' || error.name === 'NotFoundError') {
                addMessage('System', 'Camera access denied. Trying with audio only...', 'system');
                
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(audioStream => {
                        console.log('Audio-only permissions granted');
                        localStream = audioStream;
                        isVideoEnabled = false;
                        isAudioEnabled = true;
                        updateVideoButtonState();
                        updateAudioButtonState();
                        
                        // Set up audio recording
                        setupAudioRecording();
                        
                        // Start the interview
                        startInterview();
                    })
                    .catch(audioError => {
                        console.error('Error accessing audio devices:', audioError);
                        addMessage('System', 'Media access denied. Please check your permissions and refresh the page.', 'system');
                    });
            } else {
                addMessage('System', `Media access error: ${error.message}. Please check your permissions and refresh the page.`, 'system');
            }
        });
}

// Initialize speech recognition
function setupSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.error('Speech recognition not supported in this browser');
        addMessage('System', 'Speech recognition is not supported in your browser. Please use Chrome or Edge.', 'system');
        return false;
    }
    
    // Initialize the SpeechRecognition object
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    // Configure speech recognition
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US'; // Set language to English
    
    // Handle speech recognition results
    recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Update the transcription
        if (finalTranscript) {
            transcription += ' ' + finalTranscript;
            console.log('Final transcription:', transcription);
        }
        
        // Show interim results
        if (interimTranscript) {
            console.log('Interim transcription:', interimTranscript);
        }
    };
    
    // Handle speech recognition errors
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'no-speech') {
            console.log('No speech detected');
        } else {
            addMessage('System', `Speech recognition error: ${event.error}`, 'system');
        }
    };
    
    // Handle speech recognition end
    recognition.onend = () => {
        console.log('Speech recognition ended');
        recognitionActive = false;
        
        // If we were recording and have a transcription, send it
        if (isRecording && transcription.trim()) {
            stopRecordingAndSendResponse();
        }
    };
    
    console.log('Speech recognition initialized');
    return true;
}

// Start speech recognition
function startSpeechRecognition() {
    if (!recognition) {
        if (!setupSpeechRecognition()) {
            return false;
        }
    }
    
    if (!recognitionActive) {
        try {
            recognition.start();
            recognitionActive = true;
            transcription = ''; // Clear previous transcription
            console.log('Speech recognition started');
            return true;
        } catch (e) {
            console.error('Error starting speech recognition:', e);
            addMessage('System', `Error starting speech recognition: ${e.message}`, 'system');
            return false;
        }
    }
    
    return true;
}

// Stop speech recognition
function stopSpeechRecognition() {
    if (recognition && recognitionActive) {
        try {
            recognition.stop();
            recognitionActive = false;
            console.log('Speech recognition stopped');
            return true;
        } catch (e) {
            console.error('Error stopping speech recognition:', e);
            return false;
        }
    }
    
    return false;
}

// Set up audio recording
function setupAudioRecording() {
    if (!localStream || localStream.getAudioTracks().length === 0) {
        console.warn('No audio tracks for MediaRecorder');
        isAudioEnabled = false;
        updateAudioButtonState();
        return;
    }

    const audioStream = new MediaStream(localStream.getAudioTracks());
    
    // Try different MIME types based on browser support
    const mimeTypes = [
        'audio/webm',
        'audio/webm;codecs=opus',
        'audio/ogg;codecs=opus',
        'audio/mp4;codecs=opus'
    ];
    
    let options = {};
    
    // Find a supported MIME type
    for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
            options.mimeType = type;
            console.log(`Using MIME type: ${type}`);
            break;
        }
    }
    
    try {
        mediaRecorder = new MediaRecorder(audioStream, options);
        console.log('MediaRecorder created successfully');
    } catch (e) {
        console.error('MediaRecorder creation failed:', e);
        addMessage('System', `Audio recording error: ${e.message}`, 'system');
        isAudioEnabled = false;
        updateAudioButtonState();
        return;
    }

    // Variable to track silence
    let silenceTimeout = null;
    
    mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
            console.log(`Audio chunk received: ${event.data.size} bytes`);
            
            // Reset silence detection when we get data
            if (silenceTimeout) {
                clearTimeout(silenceTimeout);
            }
            
            // Set a new silence timeout
            if (isRecording) {
                silenceTimeout = setTimeout(() => {
                    console.log('Silence detected, stopping recording');
                    stopRecordingAndSendResponse();
                }, 2000); // 2 seconds of silence to stop recording
            }
        }
    };

    mediaRecorder.onstart = () => {
        console.log('MediaRecorder started');
        audioChunks = [];
        isRecording = true;
        candidateAudioIndicator.classList.add('active');
        
        // We'll start speech recognition separately after a delay
        // to avoid picking up the interviewer's audio
    };
    
    mediaRecorder.onstop = () => {
        console.log(`MediaRecorder stopped, chunks: ${audioChunks.length}`);
        if (audioChunks.length > 0) {
            const audioBlob = new Blob(audioChunks, { type: options.mimeType || 'audio/webm' });
            console.log(`Sending audio blob: ${audioBlob.size} bytes`);
            
            // Use HTTP POST instead of socket.io
            sendAudioToServer(audioBlob, transcription);
            
            // Clear for next recording
            audioChunks = [];
            candidateAudioIndicator.classList.remove('active');
        }
        isRecording = false;
        if (silenceTimeout) {
            clearTimeout(silenceTimeout);
            silenceTimeout = null;
        }
    };

    mediaRecorder.onerror = event => {
        console.error('MediaRecorder error:', event.error);
        addMessage('System', `Recording error: ${event.error.name}`, 'system');
        isRecording = false;
        candidateAudioIndicator.classList.remove('active');
    };

    // Don't auto-start recording - we'll start it after the interviewer speaks
    console.log('MediaRecorder setup complete, waiting for conversation to begin');
}

// Stop recording and send response
function stopRecordingAndSendResponse() {
    // Stop speech recognition first to get final results
    stopSpeechRecognition();
    
    // Then stop the media recorder
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        candidateAudioIndicator.classList.remove('active');
        candidateAudioIndicator.classList.remove('pulse');
        addMessage('System', 'Processing your response...', 'system');
    }
    isRecording = false;
    
    // Make sure microphone stays enabled for next turn
    if (localStream && localStream.getAudioTracks().length > 0) {
        localStream.getAudioTracks().forEach(track => {
            track.enabled = true;
        });
        isAudioEnabled = true;
        updateAudioButtonState();
    }
}

// Add message to chat
function addMessage(sender, text, type) {
    if (!chatMessages) {
        console.error('Chat messages container not found');
        return;
    }
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type);
    messageDiv.innerHTML = `<div style="font-weight: bold;">${sender}</div><div style="white-space: pre-line;">${text}</div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// End interview and cleanup
function endInterview(triggeredByButton = true) {
    if (triggeredByButton) {
        addMessage('System', 'Interview ended by user.', 'system');
    }

    // Stop candidate media
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }

    // Stop interviewer media
    if (interviewerVideo) {
        interviewerVideo.pause();
        interviewerVideo.src = '';
        interviewerVideo.style.display = 'none';
    }
    if (interviewerImage) {
        interviewerImage.style.display = 'block';
    }
    if (interviewerAudio) {
        interviewerAudio.pause();
        interviewerAudio.src = '';
    }
    updateInterviewerAudioIndicator(false);

    // Stop MediaRecorder
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }

    // Show interview finished page
    if (interviewScreen && interviewFinishedPage) {
        interviewScreen.classList.add('d-none');
        interviewFinishedPage.classList.remove('d-none');
    }

    isInterviewActive = false;
}

// Start the interview
function startInterview() {
    console.log('Starting interview...');
    isInterviewActive = true;
    
    // Ensure microphone is enabled
    if (localStream && localStream.getAudioTracks().length > 0) {
        const audioTrack = localStream.getAudioTracks()[0];
        audioTrack.enabled = true;
        isAudioEnabled = true;
        updateAudioButtonState();
    }
    
    // Display status message
    addMessage('System', 'Starting the interview...', 'system');
    
    // Make a request to start the interview
    fetch('/api/start_interview', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            candidate_name: candidateName,
            position: currentPosition
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Interview started:', data);
        handleInterviewerResponse(data);
        
        // Show instructions to the user
        setTimeout(() => {
            addMessage('System', 'The interview has started. The interviewer will ask questions, and you can respond when it\'s your turn. Your microphone will automatically activate after each question.', 'system');
        }, 1000);
    })
    .catch(error => {
        console.error('Error starting interview:', error);
        addMessage('System', `Error starting interview: ${error.message}. Please refresh and try again.`, 'system');
    });
}

// HTTP-based communication functions

// Function to send audio data to the server
async function sendAudioToServer(audioBlob, userTranscription = '') {
    try {
        if (!currentInterviewId) {
            console.error('No interview ID available');
            addMessage('System', 'Error: No active interview session. Please refresh the page.', 'system');
            return;
        }
        
        const formData = new FormData();
        formData.append('audio', audioBlob);
        formData.append('interview_id', currentInterviewId);
        
        // Include the transcription from Web Speech API if available
        if (userTranscription) {
            formData.append('transcription', userTranscription.trim());
            console.log('Sending transcription:', userTranscription.trim());
            
            // Add the user's transcription to the chat
            addMessage(candidateName, userTranscription.trim(), 'candidate');
        }
        
        const response = await fetch('/api/process_audio_message', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        handleInterviewerResponse(data);
    } catch (error) {
        console.error('Error sending audio to server:', error);
        addMessage('System', `Error processing your audio: ${error.message}`, 'system');
    }
}

// Function to send text message to the server
async function sendTextToServer(text) {
    try {
        if (!currentInterviewId) {
            console.error('No interview ID available');
            addMessage('System', 'Error: No active interview session. Please refresh the page.', 'system');
            return;
        }
        
        const response = await fetch('/api/process_text_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                text: text,
                interview_id: currentInterviewId
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        handleInterviewerResponse(data);
    } catch (error) {
        console.error('Error sending text to server:', error);
        addMessage('System', `Error processing your message: ${error.message}`, 'system');
    }
}

// Function to handle interviewer response
function handleInterviewerResponse(data) {
    if (data && data.text) {
        // Store the interview ID if provided
        if (data.interview_id && !currentInterviewId) {
            currentInterviewId = data.interview_id;
            console.log('Received interview ID:', currentInterviewId);
        }
        
        // Show the interviewer's message
        addMessage('Interviewer', data.text, 'interviewer');
        
        // Play the audio and show the video
        if (data.audio_url) {
            // Make sure any ongoing recording is stopped
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                isRecording = false;
                candidateAudioIndicator.classList.remove('active');
            }
            
            // Stop speech recognition
            stopSpeechRecognition();
            
            // Play the interviewer's response
            syncInterviewerMedia(data.audio_url);
        } else {
            console.warn('No audio_url in interviewer response');
            addMessage('System', 'Interviewer audio not available.', 'system');
        }
        
        // If there's a transcription of what the user said
        if (data.transcription) {
            addMessage(candidateName, `(You said: "${data.transcription}")`, 'candidate');
        }
        
        // If the interview is finished
        if (data.interview_finished) {
            addMessage('System', 'Interview finished. Thank you!', 'system');
            console.log('Interview finished');
            endInterview(false);
        }
    } else {
        console.warn('Invalid interviewer response:', data);
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Add pulse animation CSS
    addPulseAnimation();
    
    // Initialize media devices
    initializeMediaDevices();
    
    // Initialize interviewer media
    initializeInterviewerMedia();
    
    // Set up button event listeners
    if (toggleVideoBtn) {
        toggleVideoBtn.addEventListener('click', () => {
            if (localStream && localStream.getVideoTracks().length > 0) {
                const videoTrack = localStream.getVideoTracks()[0];
                videoTrack.enabled = !videoTrack.enabled;
                isVideoEnabled = videoTrack.enabled;
                updateVideoButtonState();
            }
        });
    }
    
    if (toggleAudioBtn) {
        toggleAudioBtn.addEventListener('click', () => {
            if (localStream && localStream.getAudioTracks().length > 0) {
                const audioTrack = localStream.getAudioTracks()[0];
                audioTrack.enabled = !audioTrack.enabled;
                isAudioEnabled = audioTrack.enabled;
                updateAudioButtonState();
                
                if (isAudioEnabled) {
                    addMessage('System', 'Microphone enabled. You can speak when it\'s your turn.', 'system');
                    
                    // If interviewer is not speaking, start recording automatically
                    if (!isInterviewerSpeaking && mediaRecorder && mediaRecorder.state !== 'recording') {
                        setTimeout(() => {
                            if (isAudioEnabled && !isInterviewerSpeaking) {
                                mediaRecorder.start();
                                isRecording = true;
                                candidateAudioIndicator.classList.add('active');
                                startSpeechRecognition();
                            }
                        }, 500);
                    }
                } else {
                    addMessage('System', 'Microphone disabled.', 'system');
                    
                    // If we're recording and audio is disabled, stop recording
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        stopRecordingAndSendResponse();
                    }
                    
                    // Stop speech recognition
                    stopSpeechRecognition();
                }
            }
        });
    }
    
    if (sendTextBtn && textInput) {
        sendTextBtn.addEventListener('click', () => {
            const text = textInput.value.trim();
            if (text) {
                addMessage(candidateName, text, 'candidate');
                // Use HTTP POST instead of socket.io
                sendTextToServer(text);
                textInput.value = '';
            }
        });
    
        textInput.addEventListener('keypress', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendTextBtn.click();
            }
        });
    }
    
    if (endCallBtnHeader) {
        endCallBtnHeader.addEventListener('click', () => {
            if (confirm('Do you really want to end the interview?')) {
                endInterview();
            }
        });
    }
    
    if (endBtn) {
        endBtn.addEventListener('click', () => {
            if (confirm('Do you really want to end the interview?')) {
                endInterview();
            }
        });
    }
    
    if (startNewInterviewBtn) {
        startNewInterviewBtn.addEventListener('click', () => {
            window.location.reload();
        });
    }
    
    if (document.getElementById('returnHomeBtn')) {
        document.getElementById('returnHomeBtn').addEventListener('click', () => {
            window.location.href = '/';
        });
    }
});

// Handle connection errors
window.addEventListener('offline', () => {
    console.error('Network connection lost');
    addMessage('System', 'Network connection lost. Please refresh when back online.', 'system');
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    endInterview(false);
});
