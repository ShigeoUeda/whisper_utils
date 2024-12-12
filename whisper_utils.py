import os
import warnings
from pydub import AudioSegment
import whisper
import torch
import gc
from typing import List, Optional

# 特定の警告を抑制
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class WhisperTranscriber:
    def __init__(self, model_size: str = "large", device: Optional[str] = None, chunk_length: int = 60):
        """
        WhisperTranscriberの初期化
        
        Args:
            model_size (str): Whisperモデルのサイズ ("tiny", "base", "small", "medium", "large")
            device (Optional[str]): 使用するデバイス。Noneの場合は自動検出
            chunk_length (int): 音声分割の長さ（秒）
        """
        self.model_size = model_size
        self.device = self._detect_device() if device is None else device
        self.chunk_length = chunk_length
        self.model = None
        self._setup_device()
        
    def _detect_device(self) -> str:
        """
        利用可能なデバイスを自動検出
        
        Returns:
            str: "cuda" または "cpu"
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _setup_device(self) -> None:
        """デバイスの設定を初期化"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = whisper.load_model(self.model_size, device=self.device)
        print("Model loaded successfully.")

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

    def transcribe_file(self, audio_path: str, output_dir: Optional[str] = None) -> str:
        """
        音声ファイルを文字起こし
        
        Args:
            audio_path (str): 音声ファイルのパス
            output_dir (Optional[str]): 出力ディレクトリ（Noneの場合は音声ファイルと同じディレクトリ）
            
        Returns:
            str: 文字起こしされたテキスト
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.model is None:
            self.load_model()

        # 出力ディレクトリの設定
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
                chunk.export(chunk_path, format="mp3")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    result = self.model.transcribe(chunk_path)
                transcript += result["text"] + "\n"
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

        # 結果の保存
        filename = os.path.basename(audio_path)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(transcript)
        
        print(f"Transcript saved to: {txt_path}")
        return transcript

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
                self.transcribe_file(audio_path, output_dir)
                print(f"Completed: {filename}")

def main():
    """
    コマンドライン引数を使用してWhisperTranscriberをテストするためのメイン関数
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
    
    # 入力パスの検証を追加
    if not args.input:
        print("Error: Input file path is empty. Please specify a valid file path.")
        return
        
    # 絶対パスに変換
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else None
    
    try:
        transcriber = WhisperTranscriber(
            model_size=args.model,
            device=args.device,
            chunk_length=args.chunk
        )
        
        if os.path.isfile(input_path):
            print(f"Processing file: {input_path}")
            transcript = transcriber.transcribe_file(input_path, output_path)
            print("Processing completed successfully.")
        elif os.path.isdir(input_path):
            print(f"Processing directory: {input_path}")
            transcriber.process_directory(input_path, output_path)
            print("Directory processing completed successfully.")
        else:
            print(f"Error: Input path does not exist: {input_path}")
            print("Please check the file path and try again.")
            return
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please make sure the file path is correct and the file exists.")
        return
    
if __name__ == '__main__':
    main()