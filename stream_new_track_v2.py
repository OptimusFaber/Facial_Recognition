import cv2, re, time, sys, os, multiprocessing
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path.
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import argparse

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
global_path = __file__
global_path = global_path.replace("stream_new_track_v2.py", "")


class Stream(multiprocessing.Process):
    def __init__(self, src, array, is_video_file, is_url):
        super(Stream, self).__init__()
        self.event = multiprocessing.Event()
        self.src = src
        self.shared_array = array
        self.counter = 0
        self.end = False
        self.is_video_file = is_video_file
        self.is_url = is_url

    def run(self):
        stream = cv2.VideoCapture(self.src)
        while True:
            ret, frame = stream.read()
            if self.is_video_file or not self.is_url:
                time.sleep(0.04)
            self.counter += 1
            if not ret:
                self.event.set()
                break
            np_array = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8).reshape(int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
            np_array[:] = frame
            # time.sleep(0.04)
        stream.release()
        return


def Save(queue, size, labels, event, save_path, fps=20):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    folder_name = datetime.today().strftime('%Y-%m-%d')
    if not os.path.exists(save_path+folder_name):
        os.mkdir(save_path+folder_name)
    label = 'NONAME'
    video_name = datetime.today().strftime(f'%Y-%m-%d-%H:%M:%S-{label}.avi')
    out = cv2.VideoWriter(save_path+folder_name+'/'+video_name, fourcc, fps, size, True)
    employee = set(labels)

    while True:
        if not queue.empty():
            status, frame, coords, labels = queue.get()
        else:
            if event.is_set():
                return
            continue
        
        if status == 1:
            if labels != [None] and coords != [()]:
                for l, c in zip(labels, coords):
                    if l is not None:
                        employee.add(l)
                        color = (0, 0, 245) if l == 'NONAME' else (0, 245, 0)
                        frame = cv2.rectangle(frame, (c[0], c[1]), (c[2],c[3]), color, 3)
                        cv2.putText(frame, l, (c[0], max(c[1]-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            out.write(frame)
        elif status == 0:
            out.release()
            employee = list(employee)
            employee.remove('NONAME')
            if employee:
                employee = '_'.join(employee)
                new_video_name = video_name.replace('NONAME', employee)
                os.rename(save_path+folder_name+'/'+video_name, save_path+folder_name+'/'+new_video_name)
            print("[SAVED]")
            break
    return


def Visualize(queue, event):
    while True:
        if not queue.empty():
            frame, labels, coords = queue.get()
            if labels != [None]:
                for l, c in zip(labels, coords):
                    if l is not None:
                        color = (0, 0, 255) if l == "NONAME" else (0, 240, 0)
                        frame = cv2.rectangle(frame, (c[0], c[1]), (c[2], c[3]), color, 3)
                        cv2.putText(frame, str(l), (c[0], c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("video_stream", frame)
            cv2.waitKey(1)
        if event.is_set():
            return



class Recognition:
    def __init__(self, input_path, model='vgg'):
        self.pics = {}
        self.img2vec = Img2Vec(cuda=True, model=model, layer_output_size=4096)
        self.model = YOLO('yolov8n-face.pt')

        for file in tqdm(os.listdir(input_path)):
            filename = os.fsdecode(file)
            img = Image.open(os.path.join(input_path, filename)).convert('RGB')
            res = self.model(img, verbose=False)
            if len(res) != 1:
                sys.exit(f'Incorrect input data: {filename}')
            boxes = res[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
            img = img.crop((x1, y1, x2, y2))
            vec = self.img2vec.get_vec(img)
            self.pics[filename] = vec

        self.bufer = {}

    def open_door(self):
        os.system("curl -i --digest -u admin:12345678elcub -X PUT -d '<RemoteControlDoor><cmd>open</cmd></RemoteControlDoor>' http://192.168.1.65/ISAPI/AccessControl/RemoteControl/door/1")


    def find(self, orig_frame):
        frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB) 
        frame = Image.fromarray(frame) 
        res = self.model.track(frame, verbose=False, device=0, persist=True)
        bounding_boxes = []
        final_labels = []
        boxes = res[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
            if box.id is not None:
                idx = int(box.id.cpu().numpy()[0].tolist())
                if self.bufer.get(idx):
                    final_labels.append(self.bufer[idx])
                    bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    continue
            frame = frame.crop((x1, y1, x2, y2))
            vec = self.img2vec.get_vec(frame)
            try:
                sims = {}
                buf = set()
                for key in list(self.pics.keys()):   
                    simularity = cosine_similarity(vec.reshape((1, -1)), self.pics[key].reshape((1, -1)))[0][0]
                    if simularity > 0.9:
                        sims[key] = simularity
                    if simularity > 0.8:
                        buf.add(re.sub('[0-9]+[.].+', '', key))
                
                buf = list(buf)
                d_view = [(v, k) for k, v in sims.items()]
                d_view.sort(reverse=True)
                name = set()
                c = 0
                for v, k in d_view:
                    c += 1
                    name.add(re.sub('[0-9]+[.].+', '', k))
                name = list(name)
                if d_view == []:
                    c = 0
                    label = 'NONAME'
                elif len(name) == 1 and len(buf) == 1:
                    label = name[0]
                    self.bufer[idx] = label
                else:
                    c = 0
                    label = 'NONAME'
                
                final_labels.append(label)
                bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            except KeyError as e:
                print('Could not find filename %s' % e)

            except Exception as e:
                print(e)
        if not boxes:
            return [None], [()]
        else:            
            return final_labels, bounding_boxes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--recognition_model", default="vgg", help="Name of recognition model")
    parser.add_argument("--source", default=None, help="the number of video camera or video stream")
    # parser.add_argument("--imgsz", default=(1920, 1080), help="video resolution")
    parser.add_argument("--visualize", default=False, help="visualize or not")
    parser.add_argument("--save_output", default=None, help="path of output file")
    parser.add_argument("--database_path", default=None, help="path to folder with images of people")

    args = parser.parse_args()
    ##----------------------------------------------------------------------------------------------
    model = args.recognition_model
    visualize = args.visualize
    ##----------------------------------------------------------------------------------------------
    if args.source is None:
        sys.exit("Video camera not found !!!")
    else:
        source = args.source
        is_video_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric()
        source = int(args.source) if webcam else args.source
        cap = cv2.VideoCapture(source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    ##-------------------------------------------------------------------------------------------
    if args.save_output is None:
        if not os.path.exists(global_path+'camera_output/'):
            os.mkdir(global_path+'camera_output/')
        output_path = global_path+'camera_output/'
    else:
        if not os.path.exists(args.save_output):
            sys.exit(f"Save dir {args.save_output} doesn't exist !!!")
        else:
            output_path = args.save_output
    ##----------------------------------------------------------------------------------------------
    if args.database_path is None:
        sys.exit("No database was provided !!!")
    else:
        if not os.path.exists(args.database_path):
            sys.exit(f"Database dir {args.save_output} doesn't exist !!!")
        else:
            db = args.database_path
    ##----------------------------------------------------------------------------------------------

    rec =Recognition(input_path = db)
    shared_array = multiprocessing.Array('B', height*width*3)

    queue = multiprocessing.Queue()
    event = multiprocessing.Event()
    if visualize:
        visualize_process = multiprocessing.Process(target=Visualize, args=(queue, event, ))
        visualize_process.start()
    # path = 'test_video.mp4'
    # path = 'example_video/zurin.avi'
    # path = '/home/elcub/experiment/example_video/burdin.avi'
    # path = "rtsp://admin:12345678elcub@192.168.1.65:554/Streaming/Channels/101"

    stream_process = Stream(source, shared_array, is_video_file, is_url)
    stream_process.start()
    queue2 = multiprocessing.Queue()

    start_recording = False
    begin = time.time()
    event2 = multiprocessing.Event()


    while stream_process:
        frame = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape((height, width, 3))
        labels, coords = rec.find(frame)
        if labels != [None]:
            begin = time.time()
            if not start_recording:
                write_process = multiprocessing.Process(target=Save, args=(queue2, (width, height), labels, event2, output_path))
                write_process.start()
                start_recording = True
                print('[START RECORDING]')
        if start_recording:
            if time.time()-begin>1.5:
                start_recording = False
                queue2.put([0, None, None, None])
            else:
                if labels != [None]:
                    queue2.put([1, frame, coords, labels])
                else:
                    queue2.put([1, frame, [()], [None]])

        if visualize:
            queue.put([frame, labels, coords])

        if stream_process.event.is_set():
            break
    print('[STREAM DIED]')
    try:
        if write_process.is_alive():
            queue2.put([0, None, None, None])
            time.sleep(0.04)
            event.set()
            event2.set()
            write_process.terminate()
    except:
        pass