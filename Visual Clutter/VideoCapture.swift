//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Video Capture for Ultralytics YOLOv8 Preview on iOS
//  Part of the Ultralytics YOLO app, this file defines the VideoCapture class to interface with the device's camera,
//  facilitating real-time video capture and frame processing for YOLOv8 model previews.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This class encapsulates camera initialization, session management, and frame capture delegate callbacks.
//  It dynamically selects the best available camera device, configures video input and output, and manages
//  the capture session. It also provides methods to start and stop video capture and delivers captured frames
//  to a delegate implementing the VideoCaptureDelegate protocol.

import AVFoundation
import CoreVideo
import UIKit
import Vision

// Defines the protocol for handling video frame capture events.
public protocol VideoCaptureDelegate: AnyObject {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame: CMSampleBuffer)
}

// Identifies the best available camera device based on user preferences and device capabilities.
func bestCaptureDevice() -> AVCaptureDevice {
  if UserDefaults.standard.bool(forKey: "use_telephoto"),
    let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: .back)
  {
    return device
  } else if let device = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: .back) {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInWideAngleCamera, for: .video, position: .back)
  {
    return device
  } else {
    fatalError("Expected back camera device is not available.")
  }
}

public class VideoCapture: NSObject {
    public var previewLayer: AVCaptureVideoPreviewLayer?
    public weak var delegate: VideoCaptureDelegate?

    let captureDevice = bestCaptureDevice()
    let captureSession = AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    var cameraOutput = AVCapturePhotoOutput()
    let queue = DispatchQueue(label: "camera-queue")

    // Configures the camera and capture session with optional session presets.
    public func setUp(
        sessionPreset: AVCaptureSession.Preset = .hd1280x720, completion: @escaping (Bool) -> Void
    ) {
        queue.async {
            let success = self.setUpCamera(sessionPreset: sessionPreset)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }

    // Internal method to configure camera inputs, outputs, and session properties.
    private func setUpCamera(sessionPreset: AVCaptureSession.Preset) -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset

        // Setup video input
        guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
            return false
        }

        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }

        // Setup video preview layer
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoOrientation = .portrait
        self.previewLayer = previewLayer

        // Setup video output
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]

        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        // Setup camera output (photo output)
        if captureSession.canAddOutput(cameraOutput) {
            captureSession.addOutput(cameraOutput)
        }

        // Set video orientation
        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
        }

        do {
            try captureDevice.lockForConfiguration()
            captureDevice.focusMode = .continuousAutoFocus
            captureDevice.exposureMode = .continuousAutoExposure
            captureDevice.unlockForConfiguration()
        } catch {
            print("Unable to configure the capture device.")
            return false
        }

        // Commit configuration to finalize setup
        captureSession.commitConfiguration()

        return true
    }

    // Starts the video capture session.
    public func start() {
        queue.async {
            if !self.captureSession.isRunning {
                self.captureSession.startRunning()
                DispatchQueue.main.async {
                    print("Camera session started.")
                }
            }
        }
    }

    // Stops the video capture session.
    public func stop() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    func updateVideoOrientation() {
        guard let connection = videoOutput.connection(with: .video) else { return }
        switch UIDevice.current.orientation {
        case .portrait:
            connection.videoOrientation = .portrait
        case .portraitUpsideDown:
            connection.videoOrientation = .portraitUpsideDown
        case .landscapeRight:
            connection.videoOrientation = .landscapeLeft
        case .landscapeLeft:
            connection.videoOrientation = .landscapeRight
        default:
            return
        }
        self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
    }
}

// Extension to handle AVCaptureVideoDataOutputSampleBufferDelegate events.
extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        delegate?.videoCapture(self, didCaptureVideoFrame: sampleBuffer)
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    
        // Pass the pixel buffer (video frame) to the Core ML model using Vision
        processFrame(pixelBuffer: pixelBuffer)
    }

    public func captureOutput(
        _ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Optionally handle dropped frames, e.g., due to full buffer.
    }
    
    // Method to process the frame using Vision and Core ML
    func processFrame(pixelBuffer: CVPixelBuffer) {
        // Load the Core ML model
        let model = try! yolov8n_seg(configuration: .init()).model
//        print("model loaded")
        
        //TODO: set preference for hardware
        
        /// VNCoreMLModel
        let detector = try! VNCoreMLModel(for: model)
        detector.featureProvider = ThresholdProvider()
    
        // Create a Vision request with the Core ML model
        let request = VNCoreMLRequest(model: detector) { request, error in
//            print(request.results)
            if let results = request.results as? [VNCoreMLFeatureValueObservation] {
                // Handle detected objects here
                print("handler")
                self.handleDetections(results)
            }
        }
        
        
    
        // Perform the Vision request on the pixel buffer (video frame)
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
    
    func handleDetections(_ results: [VNCoreMLFeatureValueObservation]) {
        var newBoxes: [CGRect] = []
        var newLabels: [String] = []
        
        
       let multiArray = results.first?.featureValue.multiArrayValue {

            // The MLMultiArray typically contains a 2D or 3D array, depending on the model.
            // For a simple segmentation model, it's likely to be 2D, where each entry is a class label.
            
            // Assuming the segmentation labels are stored in an MLMultiArray
            let labelMap = ["Background", "Object1", "Object2", ...] // Define your label map based on the model's classes.
            
            // Iterate through the MLMultiArray
            let height = multiArray.shape[0].intValue
            let width = multiArray.shape[1].intValue
            
            for y in 0..<height {
                for x in 0..<width {
                    let index = y * width + x
                    let labelIndex = multiArray[index].intValue
                    let label = labelMap[labelIndex]
                    print("Pixel (\(x), \(y)) is labeled as \(label)")
                }
            }
        }
        
        print(multiArray)
    }
}

if let results = request.results as? [VNCoreMLFeatureValueObservation],
   let multiArray = results.first?.featureValue.multiArrayValue {

    // The MLMultiArray typically contains a 2D or 3D array, depending on the model.
    // For a simple segmentation model, it's likely to be 2D, where each entry is a class label.
    
    // Assuming the segmentation labels are stored in an MLMultiArray
    let labelMap = ["Background", "Object1", "Object2", ...] // Define your label map based on the model's classes.
    
    // Iterate through the MLMultiArray
    let height = multiArray.shape[0].intValue
    let width = multiArray.shape[1].intValue
    
    for y in 0..<height {
        for x in 0..<width {
            let index = y * width + x
            let labelIndex = multiArray[index].intValue
            let label = labelMap[labelIndex]
            print("Pixel (\(x), \(y)) is labeled as \(label)")
        }
    }
}
