
import cv2
import os

import chessboard as cb
import hub_detector_lib as hd

def main():
  # NOTE: live mode requires the laptop camera's calib. Can still test other
  # stuff using other camera calibs but cannot rely on 3D output.
  live_mode = False

  camera = 'nihal_4k_portrait'
  #camera = 'limelight'

  target = 'hub'
  #target = 'chess'

  # Only for testing with a single limelight screenshot.
  if hd.DEBUG_LIMELIGHT_SCREENSHOT:
    camera = 'limelight'
    target = 'hub'

  user = os.environ['USER']

  dataDir = os.path.join('/Users', user, 'Home/robotx/data')
  calibDir = os.path.join(dataDir, 'calib_data')
  videosDir = os.path.join(dataDir, 'target_videos')
  outputDir = os.path.join(dataDir, 'output')

  os.makedirs(outputDir, exist_ok=True)

  if live_mode:
    videoFile = 0
    outputFile = os.path.join(outputDir, 'live_tracked.mp4')
  elif target == 'hub':
    videoFile = os.path.join(videosDir, 'upperhubvid_720.mp4')
    outputFile = os.path.join(outputDir, 'upperhubvid_tracked.mp4')
  elif target == 'chess':
    videoFile = os.path.join(videosDir, 'nihal_chess_4k_portrait.mp4')
    outputFile = os.path.join(outputDir, 'nihal_chess_4k_portrait_tracked.mp4')
  else:
    raise ValueError('Unknown target', target)

  if camera == 'nihal_4k_portrait':
    imageHeight = 720
    calibVideo = os.path.join(calibDir, 'nihal_chess_4k_portrait.mp4')
    calibImageHeight = 1280
    maxSamples = 30
    calib = cb.Calibration(calibVideo, calibImageHeight, maxSamples)
    if not calib.LoadOrCompute(finalImageHeight=imageHeight):
      print('Could not load or compute calibration.')
      return
    squareWidth = 0.9212598  # in inches, converted from 2.34cm
    rows = 6
    cols = 9
    chess_tracker = hd.ChessboardTracker(calib, squareWidth, rows, cols)
  elif camera == 'limelight':
    imageHeight = hd._imageHeight_ll
    calib = hd._calib_ll
    chess_tracker = hd.ChessboardTracker(calib, hd._chess_squareWidth, hd._chess_rows, hd._chess_cols)
  else:
    raise ValueError('Unknown camera type.')

  vid = cb.CameraSource(videoFile, imageHeight, adjust_height_by_orientation=False)
  out = cb.VideoWriter(outputFile)
  
  hub = hd.Hub(calib, height=hd._hub_height, cam_height=hd._cam_height, cam_pitch=hd._cam_pitch)

  while True:
    if hd.DEBUG_LIMELIGHT_SCREENSHOT:
      frame = cv2.imread(os.path.join(videosDir, 'image1.png'))
    else:
      frame = vid.GetFrame()
    if frame is None:
      break

    _, out_frame, _ = hd.runPipeline(frame, None, hub, chess_tracker, target)

    cv2.imshow('frame', out_frame)
    out.OutputFrame(out_frame)

    k = cv2.waitKey(1) & 0xFF

    if k == ord("q"):
        break

  # avg color in contour, avoid saturation, most pixels the same color (> value)



if __name__ == '__main__':
  main()
