# whisper_utils
Whisperで自動音声認識を行うライブラリ

## 環境設定

```sh
sudo apt update
sudo apt install -y python3-pip ffmpeg

git clone https://github.com/ShigeoUeda/whisper_utils.git

cd whisper_utils
python -m venv venv
source venv/bin/activate

pip install -r requairements.txt

# 上記処理で以下の作業が行われる
# pip install pydub ffmpeg git+https://github.com/openai/whisper.git

# コマンドなどの確認
whisper -h
ffmpeg --version
```

## 実行

```sh
# カレントディレクトリに音声ファイルがある場合
python whisper_utils.py -i ./audio.mp3

# 絶対パスで音声ファイルを指定する場合
python whisper_utils.py -i /full/path/to/audio.mp3

# サブディレクトリに音声ファイルがある場合
python whisper_utils.py -i ./subfolder/audio.mp3
```