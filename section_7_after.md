# 第7章後半
### この範囲をひとことで言うと：
seq2seqの使い方と改善方法（ReverseとPeeky）とseq2seqの用例紹介<br>

### 7.3.2. Decoderクラス
Encoderクラスが出力したhを受け取り、目的とする別の文字列を出力する。<br>
![alt](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/7-16.png)<br>
<br>
DecoderはRNNで実現できる。Encoderと同様にLSTMレイヤを使う。<br>
![alt](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/7-17.png)<br>
<br>
Decoderの学習時におけるレイヤ構成<br>
_ 62という教師データを使うが、このとき入力データは ['_ ', '6', '2', ' ']として与え、<br>
それに対応する出力が['6', '2', ' ', ' ']となるように学習を行う。<br>
<br>
今回の問題は「足し算」ということで、確率的な揺れを排除して「決定的」に答えを生成する（最も高いスコアを持つ文字を１つ選択する）。<br>
<br>
Decoderに文字列を生成させる流れ<br>
![alt](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/7-18.png)<br>
<br>
<br>
**argmax**: 最大値を取るインデックス（今回の例では文字ID）を選ぶノード。<br>
<br>
前節で示した文章生成のときの構成と同じ。<br>
しかし今回はSoftmaxレイヤを使わず、Affineレイヤの出力するスコアを対象に、最大の値を持つ文字IDを選ぶ<br>
<br>
Decoderでは学習時と生成時でSoftmaxレイヤの扱いが異なる。<br>
Softmax with Lossレイヤは、この後に実装するSeq2seqクラスに面倒を見てもらうことにする。<br>
→Decoderクラスは、Time Softmax with Lossレイヤの前までを担当させる<br>
<br>
![alt](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/7-19.png)<br>
<br>
Decoderクラスは、Time Embedding,Time LSTM, Time Affineの3つのレイヤから構成される<br>
[ch07/seq2seq.py](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/ch07/seq2seq.py)<br>
```
class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        #Time Embeding
        embed_W = (rn(V, D) / 100).astype('f')
        #Time LSTM
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        #Time Affine
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh
```
backward()の実装では、上方向にあるSoftmax with Lossレイヤから勾配dscoreを受け取り、<br>
Time Affine > Time LSTM > TimeEnbeddingの順に勾配を伝播させる。<br>
※Time LSTMレイヤへの時間方向への勾配は、TimeLSTMクラスのメンバ変数dhに保持されている（詳細は6.3章）。<br>
その時間方向の勾配dhを取り出し、それをDecoderクラスのbackward()の出力とする<br>
- Decoderクラスは学習時と文章生成時で挙動が異なります。
- forward()メソッドは学習時に使用されることを想定している。
- Decoderクラスに文章生成を行うメソッドをgenerate()として実装
```
    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled
```
generateは引数を3つ取る。
- **h** :Encoderから受け取る隠れ状態
- **start_id** :最初に与える文字ID
- **sample_size** :生成する文字数
文字をひとつずつ与え、Affineレイヤが出力するスコアから最大値を持つ文字IDを選ぶ作業を繰り返し行う。<br>
### 7.3.3 Seq2seqクラス
EncoderクラスとDecoderクラスをつなぎ合わせ、そしてTime Softmax with Lossレイヤを使って損失を計算する。<br>
[ch07/seq2seq.py](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/ch07/seq2seq.py)<br>
```
class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        #つなぐ
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        #損失計算
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
```
EncoderとDecoderの各クラスで、メインとなる処理はすでに実装されているため、ここではつなぎ合わせるだけ。

### 7.3.4 seq2seqの評価
seq2seqの学習は、基本的なニューラルネットワークの学習と同じ流れで行われる。
- 1.学習データからミニバッチを選ぶ
- 2.ミニバッチから勾配を計算する
- 3.勾配を使ってパラメータを更新する
<br>ここでは1.4.4 Trainerクラスで説明したTrainerクラスを使って、上の作業を行わせる<br>
またここではエポックごとにseq2seqにテストデータを解かせ(文字列生成を行わせ)、その正解率を計測する<br>
[ch07/train_seq2seq.py](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/ch07/train_seq2seq.py)<br>
```
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

# データセットの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 16
hideen_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# モデル / オプティマイザ / トレーナーの生成
model = Seq2seq(vocab_size, wordvec_size, hideen_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        # 正解率を計測
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))
```
基本的にはニューラルネットワークの学習用のコードと同じだが、ここでは評価指標として以下を採用する<br>
<br>
**正解率　－いくつの問題に正解できたか－**<br>
具体的にはエポックごとにテストデータにある問題の中でいくつかの問題に正しく答えられたかを採点する。<br>
正解率を計測するためにcommon/util.pyにあるeval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse)というメソッドを利用している<br>
※このメソッドは、問題をモデルに与えて文字列生成を行わせ、それが答えと合っているかどうかを判定する。<br>
　モデルの出す答えが合っていれば1を返し、間違っていれば0を返す。<br>
 <br>
こんな結果が表示。（acc 7.720%）<br> 
![alt](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/normal.png)<br>
 
## 7.4 seq2seqの改良
学習の進みを改善する。
### 7.4.1 入力データの反転(Reverse)
- 57+5   →   5+75
- 628+521 → 125+826
- 220 + 8 → 8 + 022
<br>
学習用のコードにデータセットを読みこみ、コードを追加（サンプルコード参照）<br>

```
# is_reverse = FalseをTrueに変更
is_reverse = True  # 
```
<br>
![alt](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/reverse.png)<br>
koyaman環境では最終的にacc 54.080%になった<br>
<br>
改善する理由は論理的ではないが勾配の伝播がスムーズになるのが理由っぽい
- 「吾輩は猫である」→「I am a cat」
- 「ある　で　猫　は　吾輩」→「I am a cat」
<br>
※吾輩とIが隣同士になるため距離が近くなる。

### 7.4.2 覗き見(Peeky)
Encoderに再度注目。Encoderは入力分を固定長のベクトルhに変換するが、LSTMだけがhを使っているのでもっと使うように活用する。<br>
こんな感じにhを活用（AffineレイヤとLSTMレイヤにhを与える）図7-26<br>
![alt](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/7-26.png)<br>
<br>
2つのベクトルが入力される場合、結合されたものになる。<br>
[ch07/peeky_seq2seq.py](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/ch07/peeky_seq2seq.py)<br>
PeekyDecoderの初期化はDecoderとほとんど同じ<br>
LSTMレイヤの重みとAffineレイヤの重みの形状が異なる。<br>
```
class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        # LSTMレイヤの重みと形状が異なる
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        # Affineレイヤの重みの形状が異なる
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None
```
forward()の実装
```
    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        # 時系列分複製
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        # 連結
        out = np.concatenate((hs, out), axis=2)
        
        out = self.lstm.forward(out)
        # 連結
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score
```
- hをnp.repeat()で時系列分複製し、それをhsにする。
- hsをEmbeddingレイヤの出力とnp.concatenate()で連結
- 連結したものをLSTMレイヤの入力にする
<br>
※Affineレイヤでも同様にする。<br>
<br>
PeekySeq2seqはSeq2seqとほぼ同様<br>
Decoderレイヤのみ異なる<br>
[ch07/peeky_seq2seq.py](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/ch07/peeky_seq2seq.py)<br>

サンプルコードtrain_seq2seq.pyのseq2seqをPeelySeq2seqに変更
```
# L33とL34のコメントアウトを入れ替え
# model = Seq2seq(vocab_size, wordvec_size, hideen_size)
model = PeekySeq2seq(vocab_size, wordvec_size, hideen_size)
```
結果めっちゃ改善する。<br>
図7-28<br>
![alt](https://github.com/koyaman2/deep-learning-from-scratch-2/blob/master/peeky.png)<br>
※koyaman環境では最終的に97.600%になった<br>

## 7.5 seq2seqを用いたアプリケーション
seq2seqは「ある時系列データ」→「別の時系列データ」に変換する。
- **機械翻訳** :「ある言語の文章」→「別の言語の文章」
- **自動要約** :「ある長い文章」→「短い要約された文章」
- **質疑応答** :「質問」→「答え」
- **メールの自動返信** :「受け取ったメールの文章」→「返信文章」
<br>
seq2seqは2つの対になった時系列データを扱う問題に利用できる。<br>
一見seq2seqに当てはめられそうにない問題でも、入力・出力データの前処理によって適用できる場合がある

### 7.5.1 チャットボット
**「相手の発言」→「自分の発言」**<br>
対話のテキストデータがあれば、それをseq2seqに学習させることができる。<br>

### 7.5.2 アルゴリズムの学習
実験では「足し算」でやっていたが、原理的にはより高度な問題も扱うことができる。<br>
**「ソースコード」→「ソースコード」**
```
input:
  j=8584
  for x in range(8):
    j+=920
  b=(1500+j)
  print((b+7567))
Target: 25011.
```
```
input:
  j=8827
  c=(i-5347)
  print((c+8704) if 2641<8500 else 5308)
 Target: 12184
```
ソースコードも文字で書かれた時系列データであり、何行にもわたるコードであっても、１つの文として処理することができる。<br>

### 7.5.3 イメージキャプション
**「画像」→「文章」**
EncoderがLSTMからCNN(Convolutional Neural Network)に置き換わっただけ。<br>
Decoderは変わらない<br>
画像のEncodeをCNNが行う<br>
CNNの出力は特徴マップ（3次元=高さ・幅・チャンネル）。<br>
これをLSTMが処理できるように1次元にする<br>
割とイケてる<br>

## まとめ
- RNNによる文章生成がテーマだった。
- 後半はseq2seqで足し算を学習させることをした。
- seq2seqはEncoderとDecoderを連結したモデルで、2つのRNNを組み合わせた単純な構造。
- seq2seqはReverseとPeekyで改良した。
