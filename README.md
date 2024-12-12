# whisper_utils
Whisperで自動音声認識を行うライブラリ

```sh
sudo apt update
sudo apt install -y python3-pip ffmpeg

git clone https://github.com/ShigeoUeda/whisper_utils.git

cd whisper_utils
python -m venv venv
source venv/bin/activate

pip install -r requairements.txt

# 以下の作業が行われる
# pip install pydub ffmpeg git+https://github.com/openai/whisper.git

# コマンドなどの確認
whisper --version
ffmpeg --version
```

