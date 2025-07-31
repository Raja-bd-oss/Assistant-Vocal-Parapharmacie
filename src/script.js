const recordBtn = document.getElementById('recordBtn');
const transcriptDiv = document.getElementById('transcript');
const responseDiv = document.getElementById('response');
const speakBtn = document.getElementById('speakResponse');
const loader = document.getElementById('loader');
const darkSwitch = document.getElementById('darkModeSwitch');
const audioUpload = document.getElementById('audioUpload');
const historyList = document.getElementById('historyList');

speakBtn.disabled = true;

const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'fr-FR';

recordBtn.addEventListener('click', () => {
  recognition.start();
  transcriptDiv.textContent = " Écoute en cours...";
  responseDiv.textContent = "";
  speakBtn.disabled = true;
  if (loader) loader.style.display = 'block';
});

recognition.onresult = (event) => {
  const text = event.results[0][0].transcript;
  transcriptDiv.textContent = text;
  getAIResponse(text);
};

recognition.onend = () => {
  if (loader) loader.style.display = 'none';
};

async function transcribeAudio(file) {
  const formData = new FormData();
  formData.append("audio", file);

  try {
    const response = await fetch("http://localhost:5000/transcribe", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Erreur serveur: ${response.status}`);
    }

    const data = await response.json();
    console.log("Réponse du serveur:", data);

    if (data.transcription) {
      document.getElementById("transcription").innerText = data.transcription;
    } else {
      throw new Error("Aucune transcription trouvée.");
    }

  } catch (error) {
    console.error("Erreur lors de la transcription :", error);
    alert("Erreur lors de la transcription : " + error.message);
  }
}


async function getAIResponse(question) {
  if (!question) return;
  if (loader) loader.style.display = 'block';

  try {
    const res = await fetch('http://localhost:5000/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });

    const data = await res.json();
    let answer = "Aucune réponse générée.";

    if (res.ok && data.answer?.trim()) {
      answer = data.answer.trim();
    } else if (data.error) {
      answer = ` Erreur backend: ${data.error}`;
    }

    responseDiv.textContent = answer;
    speakBtn.disabled = answer.startsWith("❌");

    addToHistory(question, answer);
  } catch (err) {
    console.error("Erreur /ask :", err);
    responseDiv.textContent = " Erreur: " + err.message;
    speakBtn.disabled = true;
  } finally {
    if (loader) loader.style.display = 'none';
  }
}

speakBtn.addEventListener('click', () => {
  const utterance = new SpeechSynthesisUtterance(responseDiv.textContent);
  utterance.lang = 'fr-FR';
  window.speechSynthesis.speak(utterance);
});

function addToHistory(question, response) {
  const li = document.createElement('li');
  const textSpan = document.createElement('span');
  textSpan.textContent = `Q: ${question}`;

  const playBtn = document.createElement('button');
  playBtn.textContent = "🔊";
  playBtn.title = "Lire la réponse";
  playBtn.onclick = () => {
    const utterance = new SpeechSynthesisUtterance(response);
    utterance.lang = 'fr-FR';
    window.speechSynthesis.speak(utterance);
  };

  li.appendChild(textSpan);
  li.appendChild(playBtn);
  historyList.prepend(li);
}

audioUpload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file || !file.type.startsWith("audio/")) {
    alert("Veuillez sélectionner un fichier audio valide.");
    return;
  }

  const formData = new FormData();
  formData.append('audio', file);

  transcriptDiv.textContent = " Transcription en cours...";
  responseDiv.textContent = "";
  speakBtn.disabled = true;
  if (loader) loader.style.display = 'block';

  try {
    const resTranscribe = await fetch('http://localhost:5000/transcribe', {
      method: 'POST',
      body: formData
    });

    const data = await resTranscribe.json();
    console.log("Réponse backend /transcribe :", data); 

    const transcription = data.transcription?.trim();

    if (!resTranscribe.ok || typeof transcription !== "string") {
      throw new Error(data.error || "Erreur de transcription.");
    }

    transcriptDiv.textContent = transcription;
    await getAIResponse(transcription);
  } catch (err) {
    console.error("Erreur /transcribe :", err);
    transcriptDiv.textContent = "❌ Échec lors de la transcription.";
    responseDiv.textContent = err.message;
    speakBtn.disabled = true;
  } finally {
    if (loader) loader.style.display = 'none';
  }
});

if (darkSwitch) {
  darkSwitch.addEventListener('change', () => {
    document.body.classList.toggle('dark-mode', darkSwitch.checked);
  });
}
