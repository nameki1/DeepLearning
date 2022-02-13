import tkinter
from PIL import Image

import nnlearn

class PaintApp(tkinter.Frame):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        # 画面全体のサイズ
        self.app.geometry("400x400")
        self.app.title("手書き数字を分類")

        # キャンバスのサイズ
        self.canvas = tkinter.Canvas(self.app, width=200, height=200, bg="white")
        # キャンバスの配置位置
        self.canvas.place(x=100,y=50)

        # 左クリックしながら移動している間は描画のイベント
        self.canvas.bind('<B1-Motion>',self.paint)
        
        # マウスを離したら変数を初期化
        self.canvas.bind('<ButtonRelease-1>',self.reset)

        self.befor_x=None
        self.befor_y=None

        # キャンバスをクリアするボタン
        self.clear_button = tkinter.Button(
            self.app,
            text='clear',
            command=self.clear_canvas
        )
        self.clear_button.place(x= 115,y=300)

        # キャンバスを保存するボタン
        self.save_button = tkinter.Button(
            self.app,
            text='save',
            command=self.save_canvas
        )
        self.save_button.place(x= 230,y=300)

        # 結果を表示するエリア
        self.result_box = tkinter.Label(text='0~9までの数字を書いてください')
        self.result_box.place(x=110, y=260)

        
    def paint(self,now):
        if self.befor_x and self.befor_y:
            # 描画の処理
            self.canvas.create_line(
                self.befor_x,self.befor_y,
                now.x,now.y,
                width=5,
                fill='black',
                capstyle="round",
                smooth=True
            )
        self.befor_x = now.x
        self.befor_y = now.y

    def reset(self, event):
        self.befor_x, self.befor_y = None, None

    def save_canvas(self):
        # PostScript形式で保存
        self.canvas.postscript(file='output.ps')
        # pngで保存し直す
        ps_image = Image.open('output.ps')
        ps_image.save('output.png')
        result = nnlearn.Check()
        self.result_box["text"] = "『" + str(result) + "』です"

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_box["text"] ='0~9までの数字を書いてください'



nnlearn.main()
        
app = tkinter.Tk()
app = PaintApp(app)
app.mainloop()
