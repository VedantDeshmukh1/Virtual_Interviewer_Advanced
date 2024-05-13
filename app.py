from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import assemblyai as aai
from scipy.io.wavfile import write as wav_write
import tempfile
from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
import random
from gtts import gTTS

AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  //console.log("Inside data:" + base64data)
  resolve(base64data.toString())

});

}
});

</script>
"""

def get_audio():
  display(HTML(AUDIO_HTML))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])

  process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='wav')
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
  )
  output, err = process.communicate(input=binary)

  riff_chunk_size = len(output) - 8
  # Break up the chunk size into four bytes, held in b.
  q = riff_chunk_size
  b = []
  for i in range(4):
      q, r = divmod(q, 256)
      b.append(r)

  # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
  riff = output[:4] + bytes(b) + output[8:]

  sr, audio = wav_read(io.BytesIO(riff))

  return audio, sr

def generate_questions(pdf_path, api_key):
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    satisfactory_responses = [
    "Great response!",
    "Well done!",
    "Excellent answer!",
    "Perfect!",
    "Impressive work!",
    "Fantastic job!",
    "Outstanding!",
    "Bravo!",
    "Superb effort!",
    "You nailed it!"
]
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Define the prompt template for question generation
    prompt_template = """
    Given the following text, generate interview questions based on the given text that represents a resume.

    Text:
    {text}

    Questions:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Initialize the Google Language Model
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    # Create the question generation chain
    question_chain = LLMChain(llm=llm, prompt=prompt)

    # Generate questions for each text chunk
    for i, text in enumerate(texts):
        response = question_chain.run(text=text.page_content)
        questions = response.split("\n")

        for question in questions:
            print(f"Question: {question}")
            
            # Use gTTS to read the question aloud
            tts = gTTS(text=question, lang='en', slow=False, tld='com', lang_check=False )
            tts.save("question.mp3")
            display(Audio("question.mp3", autoplay=True))

            # Get the audio recording from the user
            audio, sr = get_audio()

            # Save the audio as a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                wav_write(temp_audio_path, sr, audio)

            # Set up AssemblyAI API key
            aai.settings.api_key = "AssemblyAI API"
            transcriber = aai.Transcriber()

            # Transcribe the temporary audio file
            transcript = transcriber.transcribe(temp_audio_path)
            answer = transcript.text
            print(f"Transcription: {answer}")

            if answer.upper() == "SKIP":
                continue

            # Parse the answer and check if it is satisfactory
            if parse_response(answer, text.page_content, llm):
                print(random.choice(satisfactory_responses))
            else:
                print("I would have loved to know that. Try to provide a more relevant answer.")

            print()


def parse_response(response, resume_text, llm):
    # Define the prompt template for evaluating the response
    prompt_template = """
    Given the following resume text and the candidate's response, determine if the response is satisfactory and relevant to the information in the resume.

    Resume Text:
    {resume_text}

    Candidate's Response:
    {response}

    Is the response satisfactory and relevant to the resume? (Yes/No):
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["resume_text", "response"])

    # Create the evaluation chain
    evaluation_chain = LLMChain(llm=llm, prompt=prompt)

    # Evaluate the response
    evaluation = evaluation_chain.run(resume_text=resume_text, response=response)

    return "Yes" in evaluation

# Provide the path to your sample PDF file
pdf_path = "/content/Vedant_Resume_New_1.pdf"

# Provide your Google API key
api_key = "Google API key"

# Generate questions and answers
generate_questions(pdf_path, api_key)
