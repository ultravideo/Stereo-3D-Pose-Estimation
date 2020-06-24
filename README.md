## General
This software has been tested and developed on python3.6, although versions from 3.5 to 3.7 should work too.

### Instructions:
install requirements
```
python -m pip install -r requirements.txt
```
<br/>
The backend 2D pose estimation neural network used can be found here https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch, although this implementation includes all the code-files required to run the network. A pretrained model must be downloaded, available on above-mentioned github page or here https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth. Place the file in ```models/checkpoint_iter_370000.pth```. To train your own model, follow the instructions on Daniil Osokin's github page. 
<br/>
This implementation is farely modular and therefore any backend neural network can be used with minimal changes. No official instructions will be provided for this.
<br/><br/>

#### Usage
Edit ```pose3d.py``` and change the Lcam and Rcam variables camera ids or use precaptured video footage (use ```capture_test_videos.py``` to capture the videos) <br/>
To run the software:
```
python pose3d.py
```

### Performance
This version is parallized as much as Python allows. This implementation achieves on average 45fps-55fps on GTX1080 and Ryzen 3900X (stereo images at resolution of 360x270). <br/>
No other optimizations were done. The limiting factor in the pose estimation is usually the GPU, as the concurrent implementation of pose extraction doesn't require as much from the CPU. In nearly all cases the pose extraction phase was faster than the stereo inference. <br/>
On a laptop with GTX1650 MaxQ with i7-9750H the 3D pose estimation achieves an average framerate of 34 fps. <br/>
On both cases the parallelized algorithm is limited by the GPU. 

### NOTE:
If running the pose estimation on pre-recorded videos, make sure the flag ```waitforready``` is set to ```True```, especially on slower systems. This ensures no unwanted race conditions happen. 

## How to use the pose visualizer
The visualizer is tested on Unity 2018.4.21f1, but other (newer) versions should work too. <br/><br/>
The pose is sent over a websocket to Unity. <br/>

To open the project navigate to the ```PoseVisualizer/``` -folder using the Unity launcher.
<br/>

In ```pose3d.py``` set the the IP and PORT in ```PoseTransmitter(host="127.0.0.1", port=1234).``` <br/>
Set the IP and PORT in Unity scene view in ```IKRig SocketClient``` script (by default ```127.0.0.1:1234```). Change the properties in the editor, not in code, as Unity will adhere to editor properties by default. <br/>
Make sure in ```pose3d.py``` in ```run_3dpose``` the ```transmit``` -flag is set to ```True``` ```(transmit = True).``` It is recommended that ```downscale_resolution``` is also set to ```True```, as this makes the network run faster and be more robust. <br/>
Run ```pose3d.py``` and wait until ```awaiting connection...``` is displayed. Enter play -mode on Unity. You should now see your pose update in real-time on Unity. <br/>
To see the stickfigure model you have to be in Scene -view during playing and Game -view should be ignored.
