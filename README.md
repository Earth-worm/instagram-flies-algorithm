# instagram-flies-algorithm

## 名前
インスタバエアルゴリズム

## 着想
人間がSNSでいいねを獲得しようとする行動を模した進化的計算手法  
いいね数を稼ぐために動画の方針を決定するアルゴリズム  
個体: Instagrammer,Youtuber  
評価: いいね数  

## 流れ
- ランダムでinstagrammerを初期化
- いいね数を計算
- instagrammer集団をクラスタリング
- 新作動画の作成
- いいね数計算に戻る(任意の世代数繰り返す)


### instagrammerの初期化
  ```
  Instagrammer:
    いいね数
    作品の特徴ベクトル(最適化対象)
    方針決定確率(新作動画の作成で説明)
  ```

### クラスタリング
Instagrammer,Youtuberが作成した作品のカテゴリを決定するために、作品の特徴ベクトルのクラスタリングを行う  
クラスタリングにはkmeansを使用(実装が簡単だったため)  
計算するもの:
- 各Youtuberの作品のクラスタ(ジャンル)
- クラスタの移動速度
- クラスタに属するYoutuberのいいねの平均数
- クラスタ間の平均距離(変数ごと)

<img width="838" alt="image" src="https://github.com/Earth-worm/instagram-flies-algorithm/assets/54432132/24aae224-88fc-4088-a619-244fc2a49882">

### 新作動画の作成
新作動画の作成には次の三つの方針を方針決定確率によって選択する  
方針決定確率は「Instagrammerの初期化」で無作為に初期化される和が1となる三つの確率の配列  
方針決定確率はInstagrammerによって異なるため、各Instagrammerの個性を表す  

- Master
自分が得意な一つのジャンルでいいね数を稼ぐ
自分の作品のジャンルで一番を目指すことを目標として、作品を作成  
クラスタ内でいいねを稼いでいるYoutuber(いいね数のルーレット選択),クラスタ重心、クラスタの移動速度を参考に新作動画を作成  
<img width="878" alt="image" src="https://github.com/Earth-worm/instagram-flies-algorithm/assets/54432132/dc210733-afce-4709-94d1-2150f0abbc40">

- Faddist
流行を追う人と言う意味  
各世代で熱いジャンルを盛り上げる役割。メントスコーラ+自分の特徴みたいな感じで流行りを自分の作品に組み込む  
流行りのクラスタ内でいいねを稼いでいるYoutuber、クラスタ重心、クラスタの移動速度を参考に新作動画を作成  
流行りのクラスタといいねを稼いでいるYoutuberは、クラスタ内の平均いいね数、いいね数でルーレット選択  
<img width="877" alt="image" src="https://github.com/Earth-worm/instagram-flies-algorithm/assets/54432132/61df9a82-9863-4659-86cb-9e8ae5978787">


- Pionneer
開拓者の意味。高専系Youtuberなど、発想力で訳わからんジャンルを開拓し、一部のファンからいいねをもらう人  
自分の作品からパレート分布に従う乱数によってランダムに次回作を作成  
パレート分布にした理由は遠くに探索に行ってほしかったため。分布の由来が--とかそう言う意味はありません  
パレート分布のスケールは各クラスタ間の平均距離になっています  
画像はクラスタ間ユーグリット距離の乱数を生成していますが、実際は最適化する変数ごとに乱数を生成しています  
<img width="422" alt="image" src="https://github.com/Earth-worm/instagram-flies-algorithm/assets/54432132/22c85ff8-bb8f-4f96-938b-55ab503eba38"> 


## 実験
### 実験1
- 実験内容
最適化の様子を視覚化してみる  
```math
  f(x) = y*x (0<x,y<=100)
```  
の目的関数を最適化し、Instagrammerがグラフ右上に移動してく様子を確認する  

- 結果
青: 個体, 赤: クラスタ重心
みんな右上に向かって移動している！かわいい！  
![sample](https://github.com/Earth-worm/instagram-flies-algorithm/assets/54432132/95d6170e-8b98-4130-bfa9-1743e48d7048)  

## 実験2
- 実験内容
オセロAIの最適化でPSO(粒子群最適化法)と勝負!  
オセロAIはシンプルな方法で、下の図のようにマス目に価値を付与し、裏返したマスの価値の総和が最大となる手を選択してゆく。先行後攻は交代制
![osero](https://assets.st-note.com/img/1637765694119-RtirDLEJgR.png?width=2000&height=2000&fit=bounds&format=jpg&quality=85)  
範囲は[-30,30]で、学習100世代ごとに対戦を行い、1000世代(10回の対戦)での合計勝利数が多い方が勝利  
評価関数には最終的に残った石の数を使用しました  

- 結果
結果は勝ち!
10回中8回も勝ってくれました。ただ、負ける時はボロ負けなのが怖い。評価関数が最終の石の数なのでPSOは最大リターンを狙って最後にまくられたみたいな感じなのかもしれない。  
このオセロAIはmin-max法で3手先を読んだり、マス目の評価を定数ではなくオセロの試合の進行状況(おいた石の数)の関数とかにしたらもう少し顕著に差が生まれるのかもしれません  
<img width="957" alt="image" src="https://github.com/Earth-worm/instagram-flies-algorithm/assets/54432132/172b560a-3bf8-486c-a523-20e6be17efee">  



