import sys
import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import json
from ffmpeg_processor import FFmpegProcessor, detect_input_type

class App:
    def __init__(self, master):
        self.master = master
        self.proc = None
        master.title("车辆计数工具")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.yolo_var = tk.StringVar(value="yolo-coco")
        self.conf_var = tk.DoubleVar(value=0.5)
        self.thresh_var = tk.DoubleVar(value=0.3)
        self.gpu_var = tk.BooleanVar(value=False)
        self.inpsize_var = tk.IntVar(value=416)
        self.skip_var = tk.IntVar(value=0)
        self.display_var = tk.BooleanVar(value=True)
        self.fourcc_var = tk.StringVar(value="MJPG")
        frm = ttk.Frame(master, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        for i in range(12):
            frm.rowconfigure(i, weight=0)
        frm.columnconfigure(1, weight=1)
        ttk.Label(frm, text="输入视频").grid(row=0, column=0, sticky="w")
        ent_in = ttk.Entry(frm, textvariable=self.input_var)
        ent_in.grid(row=0, column=1, sticky="ew")
        ttk.Button(frm, text="浏览", command=self._browse_input).grid(row=0, column=2)
        ttk.Label(frm, text="输出视频").grid(row=1, column=0, sticky="w")
        ent_out = ttk.Entry(frm, textvariable=self.output_var)
        ent_out.grid(row=1, column=1, sticky="ew")
        ttk.Button(frm, text="保存为", command=self._browse_output).grid(row=1, column=2)
        ttk.Label(frm, text="YOLO目录").grid(row=2, column=0, sticky="w")
        ent_yolo = ttk.Entry(frm, textvariable=self.yolo_var)
        ent_yolo.grid(row=2, column=1, sticky="ew")
        ttk.Button(frm, text="选择", command=self._browse_yolo).grid(row=2, column=2)
        ttk.Label(frm, text="置信度").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(frm, from_=0.0, to=1.0, increment=0.05, textvariable=self.conf_var, format="%0.2f").grid(row=3, column=1, sticky="ew")
        ttk.Label(frm, text="阈值").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(frm, from_=0.0, to=1.0, increment=0.05, textvariable=self.thresh_var, format="%0.2f").grid(row=4, column=1, sticky="ew")
        ttk.Checkbutton(frm, text="使用GPU", variable=self.gpu_var).grid(row=5, column=0, sticky="w")
        ttk.Label(frm, text="输入尺寸").grid(row=6, column=0, sticky="w")
        ttk.Spinbox(frm, from_=256, to=608, increment=32, textvariable=self.inpsize_var).grid(row=6, column=1, sticky="ew")
        ttk.Label(frm, text="跳帧").grid(row=7, column=0, sticky="w")
        ttk.Spinbox(frm, from_=0, to=10, increment=1, textvariable=self.skip_var).grid(row=7, column=1, sticky="ew")
        ttk.Checkbutton(frm, text="显示窗口", variable=self.display_var).grid(row=8, column=0, sticky="w")
        ttk.Label(frm, text="编码器").grid(row=9, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.fourcc_var, values=("MJPG","XVID","MP4V")).grid(row=9, column=1, sticky="ew")
        btn_run = ttk.Button(frm, text="开始检测", command=self._run)
        btn_run.grid(row=10, column=0, sticky="ew")
        btn_stop = ttk.Button(frm, text="停止检测", command=self._stop)
        btn_stop.grid(row=10, column=1, sticky="ew")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(frm, textvariable=self.status_var).grid(row=10, column=2, sticky="w")
        self.text = tk.Text(frm, height=12)
        self.text.grid(row=11, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(11, weight=1)
        master.protocol("WM_DELETE_WINDOW", self._on_close)

    def _browse_input(self):
        p = filedialog.askopenfilename(filetypes=[("Video","*.mp4;*.avi;*.mkv;*.mov")])
        if p:
            self.input_var.set(p)

    def _browse_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI","*.avi"),("MP4","*.mp4")])
        if p:
            self.output_var.set(p)

    def _browse_yolo(self):
        p = filedialog.askdirectory()
        if p:
            self.yolo_var.set(p)

    def _run(self):
        if self.proc is not None:
            return
        args = [sys.executable, "yolo_video.py", "--input", self.input_var.get(), "--output", self.output_var.get(), "--yolo", self.yolo_var.get(), "--confidence", str(self.conf_var.get()), "--threshold", str(self.thresh_var.get()), "--input-size", str(self.inpsize_var.get()), "--skip-frames", str(self.skip_var.get()), "--fourcc", self.fourcc_var.get()]
        if self.gpu_var.get():
            args += ["--use-gpu", "1"]
        args += ["--display", "1" if self.display_var.get() else "0"]
        self.status_var.set("运行中")
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        for line in self.proc.stdout:
            self.text.insert(tk.END, line)
            self.text.see(tk.END)
        self.proc.wait()
        self.status_var.set("完成" if self.proc.returncode == 0 else f"退出 {self.proc.returncode}")
        self.proc = None

    def _stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _on_close(self):
        self._stop()
        self.master.destroy()

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()