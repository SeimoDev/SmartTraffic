import os
import json
import subprocess
import threading
import time
import re
from urllib.parse import urlparse

def _which(cmd):
    try:
        subprocess.check_output([cmd, '-version'], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def _parse_rate(rate):
    try:
        if '/' in rate:
            a, b = rate.split('/')
            return float(a) / float(b) if float(b) != 0 else 0.0
        return float(rate)
    except Exception:
        return 0.0

def detect_input_type(src):
    try:
        u = urlparse(src)
        s = (u.scheme or '').lower()
        p = (u.path or '').lower()
        if s.startswith('rtmp'):
            return 'rtmp'
        if p.endswith('.m3u8') or 'm3u8' in p:
            return 'hls'
        if p.endswith('.flv') and s in ('http','https'):
            return 'httpflv'
        if s in ('http','https'):
            return 'http'
        return 'file'
    except Exception:
        return 'file'

def ffprobe_metadata(src):
    if not _which('ffprobe'):
        return None
    try:
        cmd = ['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate','-of','json',src]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        data = json.loads(out.decode('utf-8', 'ignore'))
        s = (data.get('streams') or [{}])[0]
        w = int(s.get('width') or 0)
        h = int(s.get('height') or 0)
        r = _parse_rate(s.get('avg_frame_rate') or s.get('r_frame_rate') or '0')
        return {'width': w, 'height': h, 'fps': r}
    except Exception:
        return None

def recommend_bitrate(w, h):
    if h >= 2160 or w >= 3840:
        return '16000k'
    if h >= 1080 or w >= 1920:
        return '8000k'
    if h >= 720 or w >= 1280:
        return '4000k'
    return '2000k'

def validate_params(codec, width, height, fps, bitrate):
    okc = codec in ('libx264','h264','h264_nvenc')
    okw = isinstance(width, int) and width > 0
    okh = isinstance(height, int) and height > 0
    okf = isinstance(fps, (int,float)) and fps > 0
    okb = isinstance(bitrate, str) and re.match(r'^\d+(k|M)$', bitrate)
    return okc and okw and okh and okf and bool(okb)

class FFmpegProcessor:
    def __init__(self, source, output, codec='libx264', width=None, height=None, fps=None, bitrate=None):
        self.source = source
        self.output = output
        self.codec = codec
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.proc = None
        self.thread = None
        self.stop_flag = False
        self.status_cb = None

    def set_status_callback(self, cb):
        self.status_cb = cb

    def build_cmd(self):
        t = detect_input_type(self.source)
        in_opts = []
        if t in ('http','hls','httpflv'):
            in_opts += ['-reconnect','1','-reconnect_streamed','1','-reconnect_at_eof','1','-reconnect_delay_max','2']
        if t == 'rtmp':
            in_opts += ['-fflags','+nobuffer']
        vfilter = []
        if self.width and self.height:
            vfilter.append(f'scale={self.width}:{self.height}')
        vf = []
        if vfilter:
            vf = ['-vf','{}'.format(','.join(vfilter))]
        vopts = ['-c:v', self.codec]
        if self.fps:
            vopts += ['-r', str(self.fps)]
        if self.bitrate:
            vopts += ['-b:v', self.bitrate]
        out_opts = []
        ext = os.path.splitext(self.output)[1].lower()
        if ext == '.mp4':
            out_opts += ['-movflags','+faststart']
        cmd = ['ffmpeg','-loglevel','warning'] + in_opts + ['-i', self.source] + vf + vopts + out_opts + ['-y', self.output]
        return cmd

    def start(self):
        if not _which('ffmpeg'):
            if self.status_cb:
                self.status_cb('ffmpeg_not_found')
            return
        if self.proc is not None:
            return
        self.stop_flag = False
        def run():
            backoff = 1
            while not self.stop_flag:
                meta = ffprobe_metadata(self.source)
                if meta:
                    if self.width is None or self.height is None:
                        self.width = meta['width'] or self.width
                        self.height = meta['height'] or self.height
                    if self.fps is None:
                        self.fps = meta['fps'] or self.fps
                    if self.bitrate is None:
                        self.bitrate = recommend_bitrate(self.width or 1280, self.height or 720)
                if not validate_params(self.codec, self.width or 1280, self.height or 720, self.fps or 25, self.bitrate or '4000k'):
                    if self.status_cb:
                        self.status_cb('invalid_params')
                    return
                cmd = self.build_cmd()
                try:
                    self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                    if self.status_cb:
                        self.status_cb('started')
                    for line in self.proc.stdout:
                        if self.status_cb:
                            self.status_cb(line.strip())
                        if self.stop_flag:
                            break
                    rc = self.proc.wait()
                    self.proc = None
                    if self.stop_flag:
                        break
                    if self.status_cb:
                        self.status_cb(f'exit:{rc}')
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                except Exception as e:
                    if self.status_cb:
                        self.status_cb('error')
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        try:
            if self.proc is not None:
                self.proc.terminate()
        except Exception:
            pass

    def save_config(self, path):
        cfg = {
            'source': self.source,
            'output': self.output,
            'codec': self.codec,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'bitrate': self.bitrate,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.source = cfg.get('source') or self.source
        self.output = cfg.get('output') or self.output
        self.codec = cfg.get('codec') or self.codec
        self.width = cfg.get('width') or self.width
        self.height = cfg.get('height') or self.height
        self.fps = cfg.get('fps') or self.fps
        self.bitrate = cfg.get('bitrate') or self.bitrate