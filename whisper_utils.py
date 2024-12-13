import os
import warnings
from pydub import AudioSegment
import whisper
import torch
import gc
from typing import List, Optional

# 警告の抑制
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning)

# PyTorchのメモリ管理設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

class WhisperTranscriber:
    def __init__(self, model_size: str = "large", device: Optional[str] = None, chunk_length: int = 60):
        """
        WhisperTranscriberの初期化
        """
        self.model_size = model_size
        self.device = self._detect_device() if device is None else device
        self.chunk_length = chunk_length
        self.model = None
        self._setup_device()
        
    def _detect_device(self) -> str:
        """利用可能なデバイスを自動検出"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 8 * 1024 * 1024 * 1024:
                print("Warning: Limited GPU memory detected. Using CPU instead.")
                return "cpu"
            return "cuda"
        return "cpu"

    def _setup_device(self) -> None:
        """デバイスの設定を初期化"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            available_memory = torch.cuda.get_device_properties(0).total_memory
            if self.model_size == "large" and available_memory < 8 * 1024 * 1024 * 1024:
                print("Warning: Insufficient GPU memory for large model. Switching to medium model.")
                self.model_size = "medium"
        print(f"Using device: {self.device}")

    def _release_memory(self) -> None:
        """メモリを解放"""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def load_model(self) -> None:
        """Whisperモデルをロード"""
        self._release_memory()
        print(f"Loading Whisper {self.model_size} model...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.model = whisper.load_model(
                    self.model_size, 
                    device=self.device,
                    download_root=None,
                    in_memory=True
                )
            print("Model loaded successfully.")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU memory error during model loading. Switching to CPU...")
                self.device = "cpu"
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    self.model = whisper.load_model(
                        self.model_size, 
                        device=self.device,
                        download_root=None,
                        in_memory=True
                    )
                print("Model loaded successfully on CPU.")

    def _split_audio(self, audio_path: str) -> List[AudioSegment]:
        """音声ファイルを分割"""
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        for i in range(0, len(audio), self.chunk_length * 1000):
            chunks.append(audio[i:i + self.chunk_length * 1000])
        return chunks

    def transcribe(self, audio_path: str, output_dir: Optional[str] = None) -> str:
        """
        音声ファイルを文字起こし
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.model is None:
            self.load_model()

        if output_dir is None:
            output_dir = os.path.dirname(audio_path)
        os.makedirs(output_dir, exist_ok=True)

        chunks = self._split_audio(audio_path)
        transcript = ""
        temp_dir = os.path.dirname(audio_path)

        print(f"Processing audio file: {audio_path}")
        print(f"Total chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            chunk_path = os.path.join(temp_dir, f"temp_chunk_{i}.mp3")
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                chunk.export(chunk_path, format="mp3")
                result = self.model.transcribe(chunk_path)
                transcript += result["text"] + "\n"
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU memory error on chunk {i}. Switching to CPU...")
                    self._release_memory()
                    self.device = "cpu"
                    self.model = whisper.load_model(self.model_size, device=self.device)
                    result = self.model.transcribe(chunk_path)
                    transcript += result["text"] + "\n"
                else:
                    raise e
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                gc.collect()

        filename = os.path.basename(audio_path)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(transcript)
        
        print(f"Transcript saved to: {txt_path}")
        return transcript

    def process_directory(self, audio_dir: str, output_dir: Optional[str] = None) -> None:
        """ディレクトリ内の全音声ファイルを処理"""
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Directory not found: {audio_dir}")

        if output_dir is None:
            output_dir = audio_dir
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(audio_dir):
            if filename.lower().endswith((".mp3", ".wav", ".m4a")):
                audio_path = os.path.join(audio_dir, filename)
                print(f"\nProcessing: {filename}")
                try:
                    self.transcribe(audio_path, output_dir)
                    print(f"Completed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Whisper音声文字起こしユーティリティ')
    parser.add_argument('--input', '-i', required=True, 
                      help='入力音声ファイルまたはディレクトリのパス')
    parser.add_argument('--output', '-o', 
                      help='出力ディレクトリのパス（指定がない場合は入力と同じディレクトリ）')
    parser.add_argument('--model', '-m', default='large',
                      choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help='使用するWhisperモデルのサイズ（デフォルト: large）')
    parser.add_argument('--device',
                      choices=['cuda', 'cpu'],
                      help='使用するデバイス（未指定の場合は自動検出）')
    parser.add_argument('--chunk', '-c', type=int, default=60,
                      help='音声分割の長さ（秒）（デフォルト: 60）')
    
    args = parser.parse_args()
    
    try:
        transcriber = WhisperTranscriber(
            model_size=args.model,
            device=args.device,
            chunk_length=args.chunk
        )
        
        if os.path.isfile(args.input):
            print(f"Processing file: {os.path.abspath(args.input)}")
            transcriber.transcribe(args.input, args.output)
            print("Processing completed successfully.")
        elif os.path.isdir(args.input):
            print(f"Processing directory: {os.path.abspath(args.input)}")
            transcriber.process_directory(args.input, args.output)
            print("Directory processing completed successfully.")
        else:
            print(f"Error: Input path {args.input} does not exist.")
            return

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please make sure the file path is correct and the file exists.")

if __name__ == '__main__':
    main()