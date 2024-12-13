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
warnings.filterwarnings("ignore", category=UserWarning)

# PyTorchのメモリ管理設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

class WhisperTranscriber:
   def __init__(self, model_size: str = "large", device: Optional[str] = None, chunk_length: int = 60):
       """
       WhisperTranscriberの初期化
       
       Args:
           model_size (str): Whisperモデルのサイズ ("tiny", "base", "small", "medium", "large")
           device (Optional[str]): 使用するデバイス。Noneの場合は自動検出
           chunk_length (int): 音声分割の長さ（秒）
       """
       # 初期化時にGPUメモリを完全にクリア
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
           gc.collect()
           # メモリ使用制限を緩める（90%まで使用可能に）
           torch.cuda.set_per_process_memory_fraction(0.9)

       self.model_size = model_size
       self.device = self._detect_device() if device is None else device
       self.chunk_length = chunk_length
       self.model = None
       self._setup_device()

   def _detect_device(self) -> str:
       """利用可能なデバイスを自動検出"""
       if torch.cuda.is_available():
           return "cuda"
       return "cpu"

   def _setup_device(self) -> None:
       """デバイスの設定を初期化"""
       if self.device == "cuda":
           torch.cuda.empty_cache()
           # メモリ制限を緩和（6GB）
           torch.backends.cuda.max_memory_allocated = 6 * 1024 * 1024 * 1024
           print(f"Using device: {self.device} (Memory usage limited to 90%)")
       else:
           print(f"Using device: {self.device}")

   def _clear_gpu_memory(self) -> None:
       """GPUメモリを完全にクリア"""
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
           torch.cuda.reset_peak_memory_stats()
           gc.collect()
           print("GPU memory cleared")

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
       self._clear_gpu_memory()  # 追加：メモリクリアを徹底
       initial_device = self.device
       initial_model = self.model_size
       print(f"Attempting to load Whisper {self.model_size} model on {self.device.upper()}...")
       
       try:
           # GPU使用量制限を一時的に緩和
           if self.device == "cuda":
               torch.cuda.set_per_process_memory_fraction(0.9)
           
           with warnings.catch_warnings():
               warnings.filterwarnings("ignore", category=FutureWarning)
               warnings.filterwarnings("ignore", category=UserWarning)
               self.model = whisper.load_model(
                   self.model_size, 
                   device=self.device,
                   download_root=None,
                   in_memory=True
               )
           print(f"Model loaded successfully on {self.device.upper()}")
       
       except RuntimeError as e:
           if "out of memory" in str(e):
               if self.device == "cuda":
                   self._clear_gpu_memory()  # 追加：エラー発生時もクリア
                   if self.model_size == "large":
                       print("GPU memory error with large model. Trying medium model on GPU...")
                       self.model_size = "medium"
                       self.load_model()
                   else:
                       print("GPU memory error with medium model. Switching to large model on CPU...")
                       self.device = "cpu"
                       self.model_size = initial_model
                       self.load_model()
               else:
                   raise RuntimeError(f"Unable to load {self.model_size} model on {self.device}")
           else:
               raise e

   def _split_audio(self, audio_path: str) -> List[AudioSegment]:
       """
       音声ファイルを指定した長さに分割
       
       Args:
           audio_path (str): 音声ファイルのパス
           
       Returns:
           List[AudioSegment]: 分割された音声セグメントのリスト
       """
       audio = AudioSegment.from_file(audio_path)
       chunks = []
       for i in range(0, len(audio), self.chunk_length * 1000):
           chunks.append(audio[i:i + self.chunk_length * 1000])
       return chunks

   def transcribe(self, audio_path: str, output_dir: Optional[str] = None) -> str:
       """音声ファイルを文字起こし"""
       try:
           if not os.path.exists(audio_path):
               raise FileNotFoundError(f"Audio file not found: {audio_path}")

           if self.model is None:
               self.load_model()

           if output_dir is None:
               output_dir = os.path.dirname(audio_path)
           os.makedirs(output_dir, exist_ok=True)

           print(f"Reading audio file: {audio_path}")
           try:
               audio_segments = self._split_audio(audio_path)
           except Exception as e:
               raise RuntimeError(f"Failed to read audio file: {str(e)}")

           transcript = ""
           temp_dir = os.path.dirname(audio_path)

           print(f"Processing audio file: {audio_path}")
           print(f"Total chunks: {len(audio_segments)}")

           for i, chunk in enumerate(audio_segments, 1):
               print(f"Processing chunk {i}/{len(audio_segments)}...")
               chunk_path = os.path.join(temp_dir, f"temp_chunk_{i}.mp3")
               try:
                   if self.device == "cuda":
                       torch.cuda.empty_cache()
                       memory_allocated = torch.cuda.memory_allocated(0)
                       memory_reserved = torch.cuda.memory_reserved(0)
                       print(f"GPU Memory: Allocated = {memory_allocated/1024**2:.1f}MB, "
                             f"Reserved = {memory_reserved/1024**2:.1f}MB")
                   
                   chunk.export(chunk_path, format="mp3")
                   result = self.model.transcribe(chunk_path)
                   transcript += result["text"] + "\n"
                   
               except RuntimeError as e:
                   if "out of memory" in str(e) and self.device == "cuda":
                       print(f"GPU memory error on chunk {i}, but continuing with current configuration...")
                       self._release_memory()
                       torch.cuda.empty_cache()
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
       
       except Exception as e:
           print(f"Transcription error: {str(e)}")
           raise

   def process_directory(self, audio_dir: str, output_dir: Optional[str] = None) -> None:
       """
       ディレクトリ内の全音声ファイルを処理
       
       Args:
           audio_dir (str): 音声ファイルのディレクトリ
           output_dir (Optional[str]): 出力ディレクトリ（Noneの場合は音声ファイルと同じディレクトリ）
       """
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