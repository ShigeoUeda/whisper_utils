"""
GPUメモリを効率的に管理しながらWhisperモデルを使用する音声文字起こしユーティリティ。
メモリ不足時には代替の実行方法を提案します。
"""
import os
import warnings
from pydub import AudioSegment
import whisper
import torch
import gc
from typing import List, Optional, NoReturn, Any, Dict

# 警告の抑制
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# PyTorchのメモリ管理設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

class WhisperTranscriber:
   """
   Whisperモデルを使用した音声文字起こしクラス
   
   Attributes:
       model_size (str): Whisperモデルのサイズ ("tiny", "base", "small", "medium", "large")
       device (str): 使用するデバイス ("cuda" または "cpu")
       chunk_length (int): 音声分割の長さ（秒）
       model (Optional[Any]): ロードされたWhisperモデルのインスタンス
   """

   def __init__(self, model_size: str = "large", device: Optional[str] = None, chunk_length: int = 60) -> None:
       """
       WhisperTranscriberの初期化
       
       Args:
           model_size: Whisperモデルのサイズ
           device: 使用するデバイス（未指定時は自動検出）
           chunk_length: 音声分割の長さ（秒）
       """
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
           gc.collect()
           torch.cuda.set_per_process_memory_fraction(0.9)

       self.model_size = model_size
       self.device = self._detect_device() if device is None else device
       self.chunk_length = chunk_length
       self.model: Optional[Any] = None
       self._setup_device()

   def _detect_device(self) -> str:
       """
       利用可能なデバイスを自動検出
       
       Returns:
           str: "cuda" または "cpu"
       """
       if torch.cuda.is_available():
           return "cuda"
       return "cpu"

   def _setup_device(self) -> None:
       """
       デバイスの初期設定
       """
       if self.device == "cuda":
           torch.cuda.empty_cache()
           print(f"Using device: {self.device}")
       else:
           print(f"Using device: {self.device}")

   def _clear_memory(self) -> None:
       """
       メモリの解放とクリーンアップを実行
       """
       if self.model is not None:
           del self.model
           self.model = None
       gc.collect()
       if self.device == "cuda":
           torch.cuda.empty_cache()

   def load_model(self) -> None:
       """
       Whisperモデルをロード
       
       Raises:
           RuntimeError: GPUメモリ不足時やモデルロード失敗時
       """
       self._clear_memory()
       print(f"Attempting to load Whisper {self.model_size} model on {self.device.upper()}...")
       
       try:
           with warnings.catch_warnings():
               warnings.filterwarnings("ignore")
               self.model = whisper.load_model(
                   self.model_size, 
                   device=self.device
               )
           print(f"Model loaded successfully on {self.device.upper()}")
       
       except RuntimeError as e:
           if "out of memory" in str(e) and self.device == "cuda":
               print("\nGPU memory error occurred while loading the large model.")
               print("\nPlease try one of the following options:")
               print("1. Run with medium model on GPU:")
               print("   python whisper_utils.py -i your_audio.mp3 -m medium")
               print("2. Run with large model on CPU (slower but more accurate):")
               print("   python whisper_utils.py -i your_audio.mp3 --device cpu")
               raise RuntimeError("GPU memory insufficient for current configuration")
           else:
               raise e

   def _split_audio(self, audio_path: str) -> List[AudioSegment]:
       """
       音声ファイルを指定された長さのチャンクに分割
       
       Args:
           audio_path: 音声ファイルのパス
       
       Returns:
           List[AudioSegment]: 分割された音声セグメントのリスト
       """
       audio = AudioSegment.from_file(audio_path)
       chunks = []
       for i in range(0, len(audio), self.chunk_length * 1000):
           chunks.append(audio[i:i + self.chunk_length * 1000])
       return chunks

   def transcribe(self, audio_path: str, output_dir: Optional[str] = None) -> str:
       """
       音声ファイルを文字起こし
       
       Args:
           audio_path: 音声ファイルのパス
           output_dir: 出力ディレクトリ（Noneの場合は入力と同じディレクトリ）
       
       Returns:
           str: 文字起こしされたテキスト
       
       Raises:
           FileNotFoundError: 音声ファイルが見つからない場合
           RuntimeError: 処理中にエラーが発生した場合
       """
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
                   chunk.export(chunk_path, format="mp3")
                   result = self.model.transcribe(chunk_path)
                   transcript += result["text"] + "\n"
                   
               except RuntimeError as e:
                   if "out of memory" in str(e) and self.device == "cuda":
                       print("\nGPU memory error occurred during transcription.")
                       print("\nPlease try one of the following options:")
                       print("1. Run with medium model on GPU:")
                       print("   python whisper_utils.py -i your_audio.mp3 -m medium")
                       print("2. Run with large model on CPU (slower but more accurate):")
                       print("   python whisper_utils.py -i your_audio.mp3 --device cpu")
                       raise
                   else:
                       raise e
               finally:
                   if os.path.exists(chunk_path):
                       os.remove(chunk_path)

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
           audio_dir: 音声ファイルのディレクトリ
           output_dir: 出力ディレクトリ（Noneの場合は入力と同じディレクトリ）
       
       Raises:
           FileNotFoundError: ディレクトリが見つからない場合
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
                   raise

def main() -> NoReturn:
   """
   メイン実行関数
   
   コマンドライン引数を解析し、適切なモードでWhisperTranscriberを実行します。
   GPU メモリ不足時には代替の実行方法を提案します。
   """
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