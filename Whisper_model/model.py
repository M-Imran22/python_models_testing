import whisper
import sounddevice as sd
import numpy as np
import noisereduce as nr


def record_audio(duration=5, fs=16000):
    print(f"\n🎙️  Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs,
                       channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording complete!\n")
    return recording.squeeze(), fs


def preprocess_audio(audio: np.ndarray, fs: int):
    """
    Convert to float32, perform simple noise reduction,
    and normalize amplitude.
    """
    # Float conversion
    audio_float = audio.astype(np.float32) / 32768.0

    # Estimate noise from first 0.5 seconds
    noise_sample = audio_float[:int(0.5 * fs)]

    # Noise reduction
    reduced = nr.reduce_noise(y=audio_float, sr=fs, y_noise=noise_sample)

    # Normalize
    reduced = reduced / np.max(np.abs(reduced))

    return reduced


def transcribe_audio_np(audio: np.ndarray, fs: int, model_size="small"):
    # Preprocess
    audio_proc = preprocess_audio(audio, fs)

    print(f"🔍 Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    print("📝 Transcribing with enhanced parameters...")
    result = model.transcribe(
        audio_proc,
        language="en",
        temperature=0.0,    # deterministic
        best_of=5,          # use beam search
        beam_size=5,        # increase beams for accuracy
        fp16=False
    )
    return result


def main():
    duration_input = input(
        "⏱️ Recording duration in seconds (default 5): ").strip()
    duration = int(duration_input) if duration_input.isdigit() else 5

    audio, fs = record_audio(duration)
    result = transcribe_audio_np(audio, fs, model_size="small")

    # Display
    transcript = result.get("text", "").strip()
    print("\n=== 📝 Transcript ===")
    print(transcript)

    print("\n=== 🔊 Segments ===")
    for seg in result.get("segments", []):
        print(f"{seg['start']:.2f}-{seg['end']:.2f}s: {seg['text'].strip()!r}")


if __name__ == "__main__":
    main()
