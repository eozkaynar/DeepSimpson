"Echo-Net Dynamic Dataset"

import os
import skimage.draw
import torchvision
import collections
import cv2

import numpy    as np
import pandas   as pd

class Echo(torchvision.datasets.VisionDataset):

    def __init__(self, root=None, split="train", mean=0., std=1., length=16, period=2, max_length=250, clips=1):
        super().__init__(root)

        self.split = split.upper()
        self.mean = mean
        self.std = std
        self.length = length
        self.period = period
        self.max_length = max_length
        self.clips = clips

         # Initialize attributes
        self.fnames     = []
        self.outcome    = []
        self.header     = []

        self.frames     = collections.defaultdict(list)  # Stores frame numbers
        self.trace      = collections.defaultdict(lambda: collections.defaultdict(list))  # Stores tracing points

        # Load dataset components
        self.load_video_labels()
        self.check_missing_videos()
        self.load_traces()  
        self.filter_videos_with_traces()

    def __getitem__(self, index):

        video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = loadvideo(video).astype(np.float32) 

        # Apply normalization
        if isinstance(self.mean, (float, int)):
           # If mean is a single value, subtract it from all pixels 
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            # If std is a single value, divide all pixels by it
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)   

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        # If the video is too short, pad it with black frames (zeros)  
        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video       = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w  = video.shape  # pylint: disable=E0633 
            


        #  Determine the starting frames for extraction  
        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)
            
        # Gather targets (Extract relevant information for training/testing)
        target = []

        # Retrieve the filename of the current video
        file_name = self.fnames[index]
        target.append(self.fnames[index])  # Append the filename to the target list

        # Retrieve the frame index for End-Diastolic (ED) and End-Systolic (ES) frames
        es_index = int(self.frames[file_name][-1])  # ES Frame: Last frame in the sorted traces
        ed_index = int(self.frames[file_name][0])   # ED Frame: First frame in the sorted traces

        # Append ED & ES frame indices to the target list
        target.append(ed_index)
        target.append(es_index)


        # Append the actual ED and ES frame images from the video
        target.append(video[:, ed_index, :, :])  # Add the End-Diastolic frame
        target.append(video[:, es_index, :, :])  # Add the End-Systolic frame

        # Retrieve the traced contour points for the End-Diastolic (ED) and End-Systolic (ES) frames
        large_trace = self.trace[file_name][self.frames[file_name][-1]]  # ED frame trace (largest frame)
        small_trace = self.trace[file_name][self.frames[file_name][0]]   # ES frame trace (smallest frame)

        # -------------------------- Large Trace (ED Frame) Processing --------------------------

        # Extract x and y coordinates from the traced contour
        x1, y1, x2, y2 = large_trace[:, 0], large_trace[:, 1], large_trace[:, 2], large_trace[:, 3]

        # Concatenate x and y coordinates to form a closed contour
        x = np.concatenate((x1[1:], np.flip(x2[1:])))  # Flip ensures the contour closes properly
        y = np.concatenate((y1[1:], np.flip(y2[1:])))

        # Create a binary mask of the traced region
        r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (video.shape[2], video.shape[3]))

        # Initialize an empty mask and set the traced region to 1
        large_trace_mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
        large_trace_mask[r, c] = 1  # Mark the traced area as foreground (1)
        target.append(large_trace_mask)  # Append the generated mask to the target list

        # Mitral valve
        mitral_large_x = x2[0] 
        mitral_large_y = y2[0]
            

        # -------------------------- Small Trace (ES Frame) Processing --------------------------

        # Extract x and y coordinates from the traced contour
        x1, y1, x2, y2 = small_trace[:, 0], small_trace[:, 1], small_trace[:, 2], small_trace[:, 3]

        # Concatenate x and y coordinates to form a closed contour
        x = np.concatenate((x1[1:], np.flip(x2[1:])))  # Flip ensures the contour closes properly
        y = np.concatenate((y1[1:], np.flip(y2[1:])))

        # Create a binary mask of the traced region
        r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (video.shape[2], video.shape[3]))

        # Initialize an empty mask and set the traced region to 1
        small_trace_mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
        small_trace_mask[r, c] = 1  # Mark the traced area as foreground (1)
        target.append(small_trace_mask)  # Append the generated mask to the target list

        # Append clinical measurements (Ejection Fraction, End-Diastolic Volume, End-Systolic Volume)
        target.append(float(self.outcome[index][self.header.index("EF")]))   #  Append EF value
        target.append(float(self.outcome[index][self.header.index("EDV")]))  #  Append EDV value
        target.append(float(self.outcome[index][self.header.index("ESV")]))  #  Append ESV value

        # Mitral valve
        mitral_small_x = x2[0]
        mitral_small_y = y2[0]
        target.append(mitral_large_x)
        target.append(mitral_large_y)
        target.append(mitral_small_x)  
        target.append(mitral_small_y)  



        
        # target = [Filename, ED Frame Index, ES Frame Index, ED Frame Image, ES Frame Image, 
        # ED Trace Mask, ES Trace Mask, EF Value, EDV Value, ESV Value,ED_mitral, ED_mitralES_mitral]
        target = tuple(target)



        # Select clips from video based on starting indices
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)

        # If only one clip is selected, return it as a single tensor 
        if self.clips == 1:
            video = video[0]
        else:
            # Stack multiple clips into a single tensor with shape (N,C,F,H,W) 
            video = np.stack(video)
    
        return video, target
    
    def __len__(self):
        return len(self.fnames)

    def load_video_labels(self):
        """Load video file names and outcome labels from 'FileList.csv'."""
        file_list_path  = os.path.join(self.root, "FileList.csv")
        data            = pd.read_csv(file_list_path)
        
        # Normalize the 'Split' column to uppercase
        data["Split"]   = data["Split"].str.upper()

        # Filter by dataset split (train/val/test/all)
        if self.split != "ALL":
            data        = data[data["Split"] == self.split]

        # Store column headers and filenames
        self.header     = data.columns.tolist()
        self.fnames     = [
            fn + ".avi" if os.path.splitext(fn)[1] == "" else fn
            for fn in data["FileName"].tolist()
        ]
        self.outcome    = data.values.tolist()

    def check_missing_videos(self):
        """Check if any video files are missing from the 'Videos' directory."""
        video_dir           = os.path.join(self.root, "Videos")
        available_videos    = set(os.listdir(video_dir))
        missing_videos      = set(self.fnames) - available_videos

        if missing_videos:
            print(f"{len(missing_videos)} videos could not be found in {video_dir}:")
            for f in sorted(missing_videos):
                print("\t", f)
            raise FileNotFoundError(f"Missing video: {os.path.join(video_dir, sorted(missing_videos)[0])}")

    def load_traces(self):
        """Load frame traces from 'VolumeTracings.csv'."""
        trace_file  = os.path.join(self.root, "VolumeTracings.csv")
        with open(trace_file) as f:
            header = f.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:

                filename, x1, y1, x2, y2, frame = line.strip().split(',')
                frame                           = int(frame)
                coords                          = (float(x1), float(y1), float(x2), float(y2))

                if frame not in self.trace[filename]:
                    self.frames[filename].append(frame)
                self.trace[filename][frame].append(coords)

        # Convert traces to numpy arrays
        for filename in self.frames:
            for frame in self.frames[filename]:
                self.trace[filename][frame] = np.array(self.trace[filename][frame])

    def filter_videos_with_traces(self):
        """Remove videos that do not have at least 2 frames with traces."""
        valid_videos = [len(self.frames[f]) >= 2 for f in self.fnames]

        # Filter filenames and outcomes
        self.fnames = [f for f, valid in zip(self.fnames, valid_videos) if valid]
        self.outcome = [o for o, valid in zip(self.outcome, valid_videos) if valid]

def loadvideo(filename:str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError
    
    # Open video
    capture         = cv2.VideoCapture(filename)

    frame_count     = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width     = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height    = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v               = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)# (F ,H, W, C) 


    # Read video frame by frame  
    for count in range(frame_count):
        ret, frame  = capture.read()# If ret is True, reading is succesful 
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame           = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB 
        v[count, :, :]  = frame

    v = v.transpose((3, 0, 1, 2)) # (C, F, H, W)   

    return v