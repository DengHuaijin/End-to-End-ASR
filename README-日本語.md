# End-to-End-ASR 
## NVIDIAのOpenSeq2Seqに基づく構築したEnd-to-End音声認識ツールキット

2020/10/23 モデルのベスクラスから書く(Model Encoder Decoder)

2020/11/17 モデルのコンパイルが実現

2020/11/25 モデルの訓練が実現

2020/12/24　モデルの評価と推測が実現（しかしWERが高い、訓練コードの問題だと考えている）

### データセット：
- LibriSpeech  [ここからダウンロード](http://www.openslr.org/12)

### End-to-Endモデル
- DeepSpeech2 [論文](https://arxiv.org/abs/1512.02595)

### モデルの訓練と評価

評価モードでは単語誤り率しか出力されていない、推測モードでは認識した文字列が出力される。

- 訓練(train): python3 run.py --config_file egs/librispeech/config/ds2_small_1gpu.py --mode train

- 評価(devlopment)：python3 run.py --config_file egs/librispeech/config/ds2_small_1gpu.py --mode mode

- 推測(test)：python3 run.py --config_file egs/librispeech/config/ds2_small_1gpu.py --mode infer --infer_output_file librispeech_infer.txt

### デモ（単発話の認識） 

