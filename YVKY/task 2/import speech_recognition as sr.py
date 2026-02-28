import speech_recognition as sr

def recognize_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Speak now...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("\nTranscription:")
        print(text)
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not request results from the service.")

def recognize_from_file(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        print("\nTranscription:")
        print(text)
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not request results from the service.")

if __name__ == "__main__":
    print("Speech Recognition System")
    print("1. Use Microphone")
    print("2. Use Audio File")
    choice = input("Choose option (1 or 2): ")

    if choice == "1":
        recognize_from_mic()
    elif choice == "2":
        path = input("Enter audio file path (wav format): ")
        recognize_from_file(path)
    else:
        print("Invalid choice")